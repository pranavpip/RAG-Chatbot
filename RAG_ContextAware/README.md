# RAG Chatbot Application🤖

## Introduction
This project implements a Context-Awarew Retrieval-Augmented Generation (RAG) chatbot using Streamlit.The chatbot is powered by the Mistral-7B-Instruct-v0.3 language model integrated with ChromaDB as vector database.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [License](#license)

## Installation
To install and set up the project, follow these steps:

1. Clone the repository.
    ```bash
    git clone https://github.com/todap/RAG.git
    ```
2. Navigate to the project directory.
    ```bash
    cd RAG
    ```
3. Install the required dependencies.
    ```bash
    pip install -r requirements.txt
    ```
4. Set your Hugging Face token in the `app.py` file:
   ```python
   HF_TOKEN = st.secrets["HF_TOKEN"]
   ```

## Usage
1. Run the main application:
    ```bash
    streamlit run app.py
    ```
    ### OR
     -deployed on streamlit:https://team-qubits.streamlit.app/
3. Interact with the chatbot via the web interface.
4. Upload documents using the Document Management section and process them for use within the chatbot.

## Features

1. **Contextual Responses**: The chatbot retrieves relevant documents from a knowledge base and uses them to provide contextual responses to user queries.
2. **Conversational History**: The chatbot maintains a conversation history, allowing it to reference and build upon previous interactions.
3. **Document Management**: The application provides a document management interface, allowing users to upload and store new documents in the knowledge base.
4. **Feedback Mechanism**: Users can provide feedback on the chatbot's responses, which is used to improve the quality of future responses.

## Dependencies
The project relies on the following major dependencies:
- `streamlit`
- `huggingface_hub`
- `langchain`
- `chromadb`

## Configuration
- Store your Hugging Face token in the Streamlit secrets file as `HF_TOKEN`.
- Additional configuration options may be found within the `app.py` file.




