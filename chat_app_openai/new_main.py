import streamlit as st
import time
import os
from new_vectorizer import EmbeddingsManager
from new_agent import ChatbotManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Document Buddy App", layout="wide")

if 'temp_pdf_path' not in st.session_state:
    st.session_state['temp_pdf_path'] = None
if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.title("📄 Chat With Your Docs")
st.markdown("Upload a PDF, generate embeddings, and chat with it using OpenAI's API.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    temp_pdf_path = "temp.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state['temp_pdf_path'] = temp_pdf_path
    st.success("📄 File Uploaded Successfully!")

if st.button("Generate Embeddings"):
    if st.session_state['temp_pdf_path']:
        embeddings_manager = EmbeddingsManager(openai_api_key=api_key)
        with st.spinner("Generating embeddings..."):
            result = embeddings_manager.create_embeddings(st.session_state['temp_pdf_path'])
        st.success(result)
        
        st.session_state['chatbot_manager'] = ChatbotManager(openai_api_key=api_key)
    else:
        st.warning("Please upload a PDF first.")

st.markdown("## Chat with Document")
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

            # response = st.session_state['chatbot_manager'].get_response(user_input)
        
        # st.chat_message("assistant").markdown(response)
        # st.session_state['messages'].append({"role": "assistant", "content": response})
            st.session_state['messages'].append({"role": "assistant", "content": full_response})
else:
    st.info("Please upload a document and generate embeddings first.")
