# **Conversational Retrieval-Augmented Generation (RAG) System**

This project implements a conversational retrieval-augmented generation system using LangChain, OpenAI, and FAISS for document retrieval and Q&A generation. It allows the user to query a locally stored document or use a LangChain API for Q&A with contextual awareness and chat history.

## **Project Structure**

- `main.py`: Uses a locally stored vector database (FAISS) for document retrieval.
- `chain.py`: Uses LangChain API for document retrieval and question-answering.

## **Installation**

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/conversational-rag.git
   cd conversational-rag

2. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
3. Set up your environment variables by creating a .env file in the root directory and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your-api-key
   VECTORSTORE_PATH=path_to_your_local_vectorstore
   DATA_PATH=path_to_your_local_data_directory