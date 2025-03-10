from openai import OpenAI
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

class ChatbotManager:
    def __init__(self, 
                 openai_api_key: str,
                 openai_model: str = "gpt-4-turbo",
                 qdrant_url: str = "http://qdrant:6333",
                 collection_name: str = "vector_db"):
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.client = QdrantClient(url=self.qdrant_url, prefer_grpc=False)

        # Initialize the Qdrant vector store
        self.db = Qdrant(
            client=self.client,
            embeddings=self.embeddings,
            collection_name=self.collection_name
        )

    def get_response(self, user_query: str) -> str:
        # Retrieve relevant documents
        relevant_docs = self.db.similarity_search(user_query, k=3)
        
        if not relevant_docs:
            return "I couldn't find relevant information in the uploaded documents."

        # Construct the context with document references
        context_sections = []
        sources = set()  # To avoid duplicate document names
        for doc in relevant_docs:
            doc_text = doc.page_content.strip()
            doc_source = doc.metadata.get("source", "Unknown Document")
            context_sections.append(f"[From {doc_source}]: {doc_text}")
            sources.add(doc_source)

        context = "\n\n".join(context_sections)
        source_info = ", ".join(sources)

        prompt = f"""
        You are an AI assistant that provides answers based on the provided documents.
        If you don't know the answer, simply state that you don't know.
        
        Context:
        {context}

        User Question:
        {user_query}
        
        Answer in a clear and concise manner.
        At the end of the response, mention the source documents used: {source_info}
        """

        response = client.chat.completions.create(model=self.openai_model,
        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}],
                  temperature=0.7,
                  stream=True)

        full_response = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content

        # Append document references at the end
        final_response = full_response + f"\n\n📄 Sources: {source_info}"
        yield final_response
