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
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_url, prefer_grpc=False
        )

        # Initialize the Qdrant vector store
        self.db = Qdrant(
            client=self.client,
            embeddings=self.embeddings,
            collection_name=self.collection_name
        )

    def get_response(self, user_query: str) -> str:
        relevant_docs = self.db.similarity_search(user_query, k=3)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
        You are an intelligent assistant that provides answers based on the given documents.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}
        
        User Question:
        {user_query}
        
        Answer in a clear and concise manner.
        """

        response = client.chat.completions.create(model=self.openai_model,
        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}],
                  temperature=0.7,
                  stream=True)
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
        # return response.choices[0].message.content.strip()