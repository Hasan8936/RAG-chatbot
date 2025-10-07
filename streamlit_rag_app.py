import streamlit as st

# Lightweight Streamlit entrypoint that ensures imports work on Streamlit Cloud
st.set_page_config(page_title="RAG Chatbot", page_icon="📚")

try:
    # Import the DocumentProcessor from the backend package
    from backend.document_processor import DocumentProcessor
except Exception:
    st.title("RAG Chatbot — startup error")
    st.error("Failed to import `backend.document_processor`. Check your dependencies and Python path. See full logs for details.")
    # Re-raise so the platform logs contain the traceback for debugging
    raise


def main():
    st.title("RAG Chatbot — Streamlit entrypoint")

    st.markdown("This lightweight entrypoint ensures the `backend.document_processor` module is imported correctly.")

    # Create the processor to verify import works at runtime
    processor = DocumentProcessor()
    st.info(f"DocumentProcessor ready (chunk_size={processor.chunk_size}, chunk_overlap={processor.chunk_overlap})")

    st.write("You can now wire the rest of the Streamlit UI here.")


if __name__ == "__main__":
    main()
