<<<<<<< HEAD
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
        import streamlit as st

        # Import the DocumentProcessor from the backend package
        from backend.document_processor import DocumentProcessor


        def main():
            st.set_page_config(page_title="RAG Chatbot", page_icon="📚")

            st.title("RAG Chatbot — Streamlit entrypoint")

            st.markdown("This lightweight entrypoint ensures the `backend.document_processor` module is imported correctly.")

            # Create the processor to verify import works at runtime
            processor = DocumentProcessor()
            st.info(f"DocumentProcessor ready (chunk_size={processor.chunk_size}, chunk_overlap={processor.chunk_overlap})")

            st.write("You can now wire the rest of the Streamlit UI here.")


        if __name__ == "__main__":
            main()
            loop = asyncio.new_event_loop()
