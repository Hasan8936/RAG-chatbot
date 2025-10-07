import streamlit as st

# Import the DocumentProcessor from the backend package
from backend.document_processor import DocumentProcessor


def main():
    st.title("RAG Chatbot — Streamlit entrypoint")

    st.markdown("This lightweight entrypoint ensures the `backend.document_processor` module is imported correctly.")

    # Create the processor to verify import works at runtime
    processor = DocumentProcessor()
    st.info(f"DocumentProcessor ready (chunk_size={processor.chunk_size}, chunk_overlap={processor.chunk_overlap})")

    st.write("You can now wire the rest of the Streamlit UI here.")


if __name__ == "__main__":
    main()
