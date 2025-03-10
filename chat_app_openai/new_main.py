import streamlit as st
import os
import time
from new_vectorizer import EmbeddingsManager
from new_agent import ChatbotManager
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Document Buddy App", layout="wide")

if 'uploaded_docs' not in st.session_state:
    st.session_state['uploaded_docs'] = []
if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = ChatbotManager(openai_api_key=api_key)
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.title("📄 Chat With Your Docs")
st.markdown("Upload multiple PDFs, generate embeddings, and chat with them using OpenAI's API.")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the directory exists

uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join("uploads", f"{uploaded_file.name}")
        
        # Save uploaded file to disk
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Add the file path to the session state for further processing
        if 'uploaded_docs' not in st.session_state:
            st.session_state['uploaded_docs'] = []
        st.session_state['uploaded_docs'].append(file_path)
        st.success(f"📄 '{uploaded_file.name}' uploaded successfully!")

if st.button("Generate Embeddings"):
    if st.session_state['uploaded_docs']:
        embeddings_manager = EmbeddingsManager(openai_api_key=api_key)
        with st.spinner("Generating embeddings for all documents..."):
            for doc_path in st.session_state['uploaded_docs']:
                doc_name = os.path.basename(doc_path)  # Extract the filename from path
                result = embeddings_manager.create_embeddings(doc_path, doc_name)  # Process each file
                st.success(result)
    else:
        st.warning("Please upload at least one PDF first.")


st.markdown("## Chat with Your Documents")
if st.session_state['chatbot_manager']:
    for msg in st.session_state['messages']:
        st.chat_message(msg['role']).markdown(msg['content'])
    
    if user_input := st.chat_input("Type your question..."):
        st.chat_message("user").markdown(user_input)
        st.session_state['messages'].append({"role": "user", "content": user_input})
        
        with st.spinner("🤖 Responding..."):
            full_response = ""
            response_container = st.chat_message("assistant")
            response_area = response_container.empty()

            response_generator = st.session_state['chatbot_manager'].get_response(user_input)
            for chunk in response_generator:
                full_response += chunk
                response_area.markdown(full_response)

            st.session_state['messages'].append({"role": "assistant", "content": full_response})
else:
    st.info("Please upload and process documents first.")
