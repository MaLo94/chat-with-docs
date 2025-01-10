# Chat with Documents using Llama-3.2:1b

This project is a Streamlit-based application that enables users to upload PDF documents, index their contents, and interact with the content using a chatbot powered by the Llama-3.2:1b language model. The application utilizes advanced embeddings and vector search techniques to deliver contextual and precise responses to user queries.

---

## Features

- **PDF Upload and Preview**: Upload PDF documents and preview their content directly in the application.
- **Document Indexing**: Automatically indexes uploaded documents using HuggingFace embeddings for efficient querying.
- **Chat Interface**: Chat with the document content using a customized prompt template for accurate and context-aware responses.
- **Session Management**: Unique session IDs and file caching for seamless interaction.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/chat-with-docs.git
   cd chat-with-docs
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Access the application in your browser at `http://localhost:8501`.

---

## How to Use

1. **Upload a Document**:
   - Use the sidebar to upload a `.pdf` file.
   - The application will index the document and display a preview.

2. **Interact with the Document**:
   - Enter your query in the chat input box.
   - The chatbot will respond based on the content of the uploaded document.

3. **Clear Session**:
   - Use the "Clear ↺" button to reset the session and start fresh.

---

## Technical Details

### Core Components
- **Llama-3.2:1b Model**: Utilized for language understanding and generating responses.
- **HuggingFace Embeddings**: Used to encode document content into vector space.
- **VectorStoreIndex**: Enables efficient storage and retrieval of indexed document content.
- **Streamlit**: Provides an intuitive user interface for document interaction.

### Custom Prompt Template
The chatbot employs a structured prompt for context-aware and concise responses:
```text
Context information is below.
---------------------
{context_str}
---------------------
Given the context information above I want you to think step by step to answer the query in a crisp manner. In case you don't know the answer, say 'I don't know!'.
Query: {query_str}
Answer:
```

---

## Roadmap

This project is ongoing, and the following features will be added soon:

1. **Logs and Storage**:
   - Implement logging of user queries and chatbot responses.
   - Store logs in MongoDB for analysis and future reference.

2. **Vector Database Integration**:
   - Integrate a vector database for storing context and embeddings for scalability.

3. **Dockerization**:
   - Dockerize the entire application for easy deployment and portability.

4. **Gateway Service**:
   - Add a FastAPI-based gateway service to manage backend operations and API integrations.

---
