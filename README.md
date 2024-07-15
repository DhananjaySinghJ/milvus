# milvus
# CUDA Documentation Search

This project provides a web application for searching through CUDA documentation using a combination of BM25 and BERT models. It uses Milvus for managing the embeddings and Streamlit for the user interface.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Detailed Function Descriptions](#detailed-function-descriptions)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The CUDA Documentation Search application allows users to search through the NVIDIA CUDA documentation efficiently. It leverages natural language processing (NLP) techniques and a hybrid retrieval system combining BM25 and BERT models to provide accurate and relevant search results.

## Features

- **Web Crawling**: Recursively crawls web pages up to a specified depth to collect data.
- **Data Clustering**: Clusters the collected data into meaningful groups using KMeans.
- **Vector Embedding**: Converts data into vector embeddings using Sentence Transformers.
- **Milvus Integration**: Stores and manages embeddings in Milvus, a highly scalable vector database.
- **Hybrid Search**: Combines BM25 and BERT for efficient and accurate search results.
- **Streamlit UI**: User-friendly interface for entering queries and viewing results.

## Installation

To set up and run the project locally, follow these steps:

### Prerequisites

- Python 3.7 or higher
- pip
- Docker (for running Milvus)

### Steps

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/cuda-docs-search.git
    cd cuda-docs-search
    ```

2. **Set up a virtual environment** (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required Python packages**:

    ```bash
    pip install pymilvus sentencetransformers sk-learn
    ```

4. **Run Milvus using Docker**:

    ```bash
    docker-compose up -d
    ```

## Usage

1. **Run the Streamlit application**:

    ```bash
    streamlit run main.py
    ```

2. **Open your browser** and navigate to `http://localhost:8501` to access the application.

3. **Enter your query** in the provided text input and hit enter to search through the CUDA documentation.



