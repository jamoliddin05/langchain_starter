import os
from glob import glob
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    data_directory = os.getenv('DATA_PATH')

    # Get a list of all .txt files in the data directory
    txt_files = glob(os.path.join(data_directory, '*.txt'))

    # Read the contents of all .txt files and store them in a list
    document_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as file:
            document = [Document(page_content=file.read())]
            document_chunks.extend(text_splitter.split_documents(document))

    token = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=token)

    vectorstore = FAISS.from_documents(document_chunks, embeddings)
    vectorstore.save_local(os.getenv('VECTORSTORE_PATH'))
