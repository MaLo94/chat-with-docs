import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

class EmbeddingsManager:
    def __init__(self, openai_api_key: str, qdrant_url: str = "http://qdrant:6333", collection_name: str = "vector_db"):
        self.openai_api_key = openai_api_key
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.client = QdrantClient(url=self.qdrant_url)

        # Ensure the collection exists or create it if not
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        """Check if the collection exists in Qdrant and create it if not."""
        collections_response = self.client.get_collections()
        collection_names = [collection.name for collection in collections_response.collections]  # Corrected
        
        if self.collection_name not in collection_names:
            print(f"⚠️ Collection '{self.collection_name}' not found. Creating it now.")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)  # Corrected creation
            )
            print(f"✅ Collection '{self.collection_name}' created successfully!")

    def create_embeddings(self, pdf_path: str, doc_name: str) -> str:
        """Processes a PDF file, extracts text, and stores embeddings in Qdrant with metadata."""
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

        # ✅ Ensure each split contains metadata
        for split in splits:
            split.metadata["source"] = doc_name  # ✅ Correct way to store metadata

        # ✅ Pass `splits` directly; each document now has its own metadata
        vector_store = Qdrant.from_documents(
            splits,
            self.embeddings,
            url=self.qdrant_url,
            collection_name=self.collection_name
        )

        return f"📄 '{doc_name}' successfully stored in Qdrant!"
