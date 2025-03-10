import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

class EmbeddingsManager:
    def __init__(self, openai_api_key: str, qdrant_url: str = "http://qdrant:6333", collection_name: str = "vector_db"):
        self.openai_api_key = openai_api_key
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

    def create_embeddings(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")
        
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            raise ValueError("No documents were loaded from the PDF.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
        splits = text_splitter.split_documents(docs)
        if not splits:
            raise ValueError("No text chunks were created from the documents.")
        
        Qdrant.from_documents(splits, self.embeddings, url=self.qdrant_url, collection_name=self.collection_name)
        
        return "Vector DB Successfully Created and Stored in Qdrant!"