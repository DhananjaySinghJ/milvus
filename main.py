import logging
import uuid
from urllib.parse import urljoin, urlparse

import numpy as np
import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from rank_bm25 import BM25Okapi

from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to check if a URL is valid
def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in ('http', 'https')

# Recursive function to crawl web pages up to a specified depth
def crawl(url, depth, max_depth, visited=set(), max_urls_per_depth=5):
    if depth > max_depth or url in visited:
        return []

    visited.add(url)
    try:
        response = requests.get(url, timeout=10)  # Fetch the page content
        response.raise_for_status()
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
        logging.error(f"Failed to retrieve {url}: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True) if is_valid_url(urljoin(url, a['href']))]
    logging.info(f"Depth: {depth}, URL: {url}, Found links: {len(links)}")
    
    data = [soup.get_text()]  # Extract text data from the page
    
    for link in links[:max_urls_per_depth]:  # Recursively crawl the links found on the page
        data.extend(crawl(link, depth + 1, max_depth, visited, max_urls_per_depth))
    
    return data

# Function to chunk data and perform clustering
def chunk_data(data, n_clusters=10):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Load the sentence transformer model
    embeddings = model.encode(data)  # Encode the data into embeddings
    kmeans = KMeans(n_clusters=n_clusters).fit(embeddings)  # Perform KMeans clustering
    
    labeled_data = list(zip(kmeans.labels_, data, embeddings))
    labeled_data.sort(key=lambda x: x[0])  # Sort data based on cluster labels
    
    sorted_labels, sorted_data, sorted_embeddings = zip(*labeled_data)
    
    return list(sorted_data), list(sorted_embeddings)

# Function to setup the Milvus collection
def setup_milvus():
    try:
        connections.connect("default", host='localhost', port='19530')

        fields = [
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False)
        ]

        schema = CollectionSchema(fields, "Document collection")

        if utility.has_collection("cuda_docs"):
            collection = Collection("cuda_docs")
            logging.info("Using existing collection 'cuda_docs'")
        else:
            collection = Collection("cuda_docs", schema=schema)
            logging.info("Created new collection 'cuda_docs'")

        if not collection.has_index():
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 1024}
            }
            collection.create_index("embedding", index_params)
            logging.info("Created index for collection 'cuda_docs'")

        collection.load()
        return collection

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise

# Function to insert chunks and embeddings into Milvus
def insert_into_milvus(collection, chunks, embeddings):
    entities = []
    
    logging.info(f"Number of chunks: {len(chunks)}")
    logging.info(f"Number of embeddings: {len(embeddings)}")
    
    if len(chunks) != len(embeddings):
        logging.error(f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})")
        return

    for chunk, embedding in zip(chunks, embeddings):
        unique_id = uuid.uuid4().int & (1<<63)-1
        
        entity = [
            embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            chunk[:65535],  # Ensure the chunk length is within the maximum length
            unique_id
        ]
        entities.append(entity)
    
    logging.info(f"Prepared {len(entities)} entities for insertion")
    
    batch_size = 1000
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i+batch_size]
        try:
            collection.insert(batch)
            logging.info(f"Inserted batch of {len(batch)} entities")
        except Exception as e:
            logging.error(f"Error inserting batch: {e}")
            logging.error(f"First entity in problematic batch: {batch[0]}")
    
    logging.info(f"Attempted to insert {len(entities)} entities into Milvus")

# Function to expand query using BM25 and BERT models
def query_expansion(query, data, bm25_model, bert_model):
    tokenized_query = query.split()
    bm25_scores = bm25_model.get_scores(tokenized_query)
    top_docs = np.argsort(bm25_scores)[-5:]  # Select top 5 documents
    expanded_query = ' '.join([data[i] for i in top_docs])
    
    embeddings = bert_model.encode([query, expanded_query])  # Encode the original and expanded query
    return embeddings

# Function to perform hybrid retrieval using BM25 and BERT models
def hybrid_retrieval(query, data, bm25_model, bert_model, collection):
    query_embeddings = query_expansion(query, data, bm25_model, bert_model)
    
    search_param = {'nprobe': 10}
    results = collection.search([query_embeddings[0].tolist()], "embedding", search_param, limit=10, output_fields=["url"])
    
    return results[0]  # Return just the hits for the first (and only) query

# Function to answer questions based on the retrieved documents
def answer_question(query, collection):
    qa_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Get all URLs from collection for context
    urls = [hit.entity.get('url', '') for hit in collection]
    context = ' '.join(urls)
    
    if not context:
        return {"answer": "Sorry, I couldn't find any relevant information to answer your question."}
    
    try:
        answer = qa_model(question=query, context=context)
        return answer
    except ValueError:
        return {"answer": "I found some information, but it wasn't sufficient to answer your question accurately."}

# Main function to run the Streamlit application
def main():
    st.title('CUDA Documentation Search')

    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

    if not st.session_state.initialized:
        with st.spinner('Initializing... This may take a few minutes.'):
            url = 'https://docs.nvidia.com/cuda/'
            data = crawl(url, 0, 5)
            
            chunks, embeddings = chunk_data(data)
            
            collection = setup_milvus()
            insert_into_milvus(collection, chunks, embeddings)
            
            tokenized_data = [doc.split() for doc in data]
            st.session_state.bm25 = BM25Okapi(tokenized_data)
            st.session_state.bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            st.session_state.collection = collection
            st.session_state.data = data

            st.session_state.initialized = True
        st.success('Initialization completed!')

    query = st.text_input('Enter your query:')
    if query:
        with st.spinner('Searching...'):
            results = hybrid_retrieval(query, st.session_state.data, st.session_state.bm25, 
                                       st.session_state.bert_model, st.session_state.collection)
            answer = answer_question(query, results)

        st.subheader("Answer:")
        st.write(answer['answer'])

        st.subheader("Top Retrieved Documents:")
        if results:
            for hit in results[:5]:  # Display top 5 results
                url = hit.entity.get('url', 'No URL available')
                st.write(f"- {url}")
        else:
            st.write("No relevant documents found.")

        # Add debug information
        st.subheader("Debug Information:")
        st.write(f"Number of results: {len(results)}")
        st.write(f"First result: {results[0].entity if results else 'None'}")

if __name__ == "__main__":
    main()
