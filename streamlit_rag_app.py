"""
Streamlit RAG Document Q&A System

A complete document question-answering system built with Streamlit.
Upload documents (PDF, DOCX, TXT) and ask questions about them using AI.
"""

import streamlit as st
import tempfile
import os
from typing import List, Dict, Any
import uuid
from datetime import datetime

# Import our existing backend modules
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .document-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .source-card {
        background-color: #e8f4f8;
        padding: 0.75rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        border-left: 3px solid #1f77b4;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .assistant-message {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .confidence-badge {
        background-color: #4caf50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.document_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        st.session_state.vector_store = VectorStore()
        st.session_state.rag_pipeline = RAGPipeline(st.session_state.vector_store)
        st.session_state.documents = {}  # {doc_id: {filename, chunks_count, timestamp}}
        st.session_state.chat_history = []  # List of {role, content, sources, timestamp}
        st.session_state.processing = False

initialize_session_state()

def process_uploaded_file(uploaded_file):
    """Process an uploaded file and add it to the vector store"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process document
        with st.spinner(f'Processing {uploaded_file.name}...'):
            chunks = st.session_state.document_processor.process_document(tmp_path, uploaded_file.name)
            
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Add to vector store (synchronous call)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(st.session_state.vector_store.add_documents(chunks, doc_id))
            
            # Store document metadata
            st.session_state.documents[doc_id] = {
                'filename': uploaded_file.name,
                'chunks_count': len(chunks),
                'timestamp': datetime.now(),
                'id': doc_id
            }
        
        # Cleanup
        os.unlink(tmp_path)
        
        return True, f"Successfully processed '{uploaded_file.name}' into {len(chunks)} chunks!"
    
    except Exception as e:
        return False, f"Error processing file: {str(e)}"

def delete_document(doc_id):
    """Delete a document from the system"""
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(st.session_state.vector_store.delete_document(doc_id))
        
        filename = st.session_state.documents[doc_id]['filename']
        del st.session_state.documents[doc_id]
        
        return True, f"Successfully deleted '{filename}'"
    except Exception as e:
        return False, f"Error deleting document: {str(e)}"

def ask_question(question: str):
    """Process a question through the RAG pipeline"""
    try:
        # Build chat history for context
        chat_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.chat_history[-6:]  # Last 6 messages
        ]
        
        # Query the RAG pipeline
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            st.session_state.rag_pipeline.query(question, chat_history)
        )
        
        # Add to chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': question,
            'timestamp': datetime.now()
        })
        
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': result['answer'],
            'sources': result['sources'],
            'confidence': result['confidence'],
            'timestamp': datetime.now()
        })
        
        return True, result
    
    except Exception as e:
        return False, str(e)

# Main App Layout
def main():
    # Header
    st.markdown('<div class="main-header">📚 RAG Document Q&A System</div>', unsafe_allow_html=True)
    st.markdown("Upload documents and ask questions about them using AI")
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File uploader
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files"
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Files", type="primary"):
                for uploaded_file in uploaded_files:
                    success, message = process_uploaded_file(uploaded_file)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                st.rerun()
        
        st.divider()
        
        # Document list
        st.subheader(f"Uploaded Documents ({len(st.session_state.documents)})")
        
        if st.session_state.documents:
            for doc_id, doc_info in st.session_state.documents.items():
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{doc_info['filename']}**")
                        st.caption(f"{doc_info['chunks_count']} chunks • {doc_info['timestamp'].strftime('%H:%M:%S')}")
                    with col2:
                        if st.button("🗑️", key=f"delete_{doc_id}", help="Delete document"):
                            success, message = delete_document(doc_id)
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                            st.rerun()
        else:
            st.info("No documents uploaded yet")
        
        st.divider()
        
        # Stats
        if st.session_state.documents:
            st.subheader("📊 Statistics")
            total_chunks = sum(doc['chunks_count'] for doc in st.session_state.documents.values())
            st.metric("Total Documents", len(st.session_state.documents))
            st.metric("Total Chunks", total_chunks)
            st.metric("Vector Store Size", st.session_state.vector_store.index.ntotal)
        
        st.divider()
        
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()
    
    # Main chat area
    if not st.session_state.documents:
        st.info("👆 Upload some documents using the sidebar to get started!")
    else:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    with st.chat_message("user"):
                        st.write(message['content'])
                        st.caption(message['timestamp'].strftime('%H:%M:%S'))
                else:
                    with st.chat_message("assistant"):
                        st.write(message['content'])
                        
                        # Display sources
                        if message.get('sources'):
                            with st.expander(f"📚 Sources ({len(message['sources'])})", expanded=False):
                                for i, source in enumerate(message['sources'], 1):
                                    st.markdown(f"""
                                    <div class="source-card">
                                        <strong>Source {i}:</strong> {source['source']}<br>
                                        <strong>Chunk:</strong> {source['chunk_id'] + 1}<br>
                                        <strong>Confidence:</strong> <span class="confidence-badge">{source['confidence']*100:.1f}%</span><br>
                                        <strong>Preview:</strong> {source['preview']}
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Overall confidence
                            if message.get('confidence'):
                                st.caption(f"Overall confidence: {message['confidence']*100:.1f}%")
                        
                        st.caption(message['timestamp'].strftime('%H:%M:%S'))
        
        # Chat input
        st.divider()
        question = st.chat_input("Ask a question about your documents...")
        
        if question:
            with st.spinner("Thinking..."):
                success, result = ask_question(question)
                if success:
                    st.rerun()
                else:
                    st.error(f"Error: {result}")

# Environment check
def check_environment():
    """Check if required environment variables are set"""
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY not found in environment variables!")
        st.info("Please set your OpenAI API key in the environment or .env file")
        st.stop()

if __name__ == "__main__":
    check_environment()
    main()
