# AI-Powered News Research Assistant

This project is an AI-powered news research tool that allows users to input news article URLs, processes the data, and enables users to ask questions about the content. The project leverages **LangChain**, **Cohere API**, **FAISS**, and **Streamlit** for efficient text processing, embeddings, and retrieval.

## Features
- Accepts news article URLs as input.
- Splits and processes text from the articles.
- Uses **Cohere API** to generate embeddings for document retrieval.
- **FAISS** is used to store and retrieve document vectors.
- Provides question answering functionality based on the processed articles using **LangChain**.
- Simple user interface built using **Streamlit**.

## Installation Instructions

### Setting up a virtual environment
1. Create a virtual environment:
    ```bash
    python3 -m venv path/to/venv
    ```
2. Activate the virtual environment:
    ```bash
    source path/to/venv/bin/activate
    ```

### Installing the required libraries

3. Install the required Python libraries:
    ```bash
    python3 -m pip install streamlit
    pip install -U langchain-community
    pip install langchain-cohere
    pip install faiss-cpu
    pip install unstructured
    ```

### Running the Application

4. Run the Streamlit app:
    ```bash
    streamlit run News_Research_Tool.py
    ```

## Usage

1. Enter the URLs of news articles in the sidebar (up to 3 URLs).
2. Click on **Process URLs** to start processing the articles.
3. After processing, enter a question in the input field to query the articles.
4. The app will display the answer along with the sources from the articles.

## Tech Stack

- **LangChain**: Used to build the question-answering chain and text splitting.
- **Cohere API**: Used to generate embeddings for document retrieval.
- **FAISS**: Fast Approximate Nearest Neighbor Search for efficient vector similarity search.
- **Streamlit**: Front-end tool to build an interactive interface for users.
- **Unstructured**: For document loading and text extraction from URLs.
