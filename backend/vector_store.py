# backend/vector_store.py

"""
This file creates a "vector store" - think of it as a super-smart library system.

Instead of organizing books by title or author, it organizes text chunks by MEANING.
It converts text into mathematical vectors (arrays of numbers) that represent the meaning.
Similar meanings have similar numbers, so we can find related content quickly.

Imagine if every book had a "DNA code" that represented its content - that's what we're doing here!
"""

import faiss  # Facebook's library for fast similarity search
import numpy as np  # For working with arrays of numbers
from sentence_transformers import SentenceTransformer  # Converts text to vectors
from typing import List, Dict, Any, Tuple  # For type hints
import pickle  # For saving/loading data
import os     # For file operations

class VectorStore:
    """
    This class is like a magical librarian that:
    1. Converts text into mathematical "fingerprints" (vectors)
    2. Stores these fingerprints in a searchable database
    3. Can quickly find similar content when you ask a question
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize our vector store
        
        Args:
            model_name: Which AI model to use for converting text to vectors
                       "all-MiniLM-L6-v2" is fast and good for most uses
        """
        # This model converts text to vectors (arrays of 384 numbers)
        self.model = SentenceTransformer(model_name)
        
        # Get the size of vectors this model creates
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Create a FAISS index - this is our searchable database
        # IndexFlatIP means "flat index with inner product" (good for similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Store the actual text content alongside the vectors
        self.documents: List[Dict[str, Any]] = []
        
        # Keep track of which chunks belong to which documents
        self.doc_to_chunks: Dict[str, List[int]] = {}
        
    def is_initialized(self) -> bool:
        """Check if we have any documents stored yet"""
        return self.index.ntotal > 0
    
    async def add_documents(self, chunks: List[Dict[str, Any]], doc_id: str):
        """
        Add document chunks to our vector store
        
        This is like:
        1. Taking each chunk of text
        2. Converting it to a mathematical fingerprint
        3. Filing it in our searchable database
        
        Args:
            chunks: List of text chunks with metadata
            doc_id: Unique identifier for this document
        """
        # Extract just the text content from each chunk
        texts = [chunk["content"] for chunk in chunks]
        
        # Convert all texts to vectors (this is the AI magic!)
        # normalize_embeddings=True makes all vectors the same length for fair comparison
        print(f"Converting {len(texts)} chunks to vectors...")
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        # Add vectors to our searchable index
        start_idx = len(self.documents)  # Where to start numbering new chunks
        self.index.add(embeddings.astype('float32'))  # FAISS likes 32-bit floats
        
        # Store document metadata and track chunk locations
        chunk_indices = []
        for i, chunk in enumerate(chunks):
            chunk_idx = start_idx + i
            chunk_indices.append(chunk_idx)
            
            # Add document ID to metadata so we know which doc this came from
            chunk["metadata"]["doc_id"] = doc_id
            self.documents.append(chunk)
        
        # Remember which chunks belong to this document (for deletion later)
        self.doc_to_chunks[doc_id] = chunk_indices
        
        print(f"Added {len(chunks)} chunks to vector store. Total chunks: {len(self.documents)}")
    
    async def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for documents similar to a query
        
        This is like asking a librarian: "Find me books similar to this topic"
        
        Args:
            query: The question or topic to search for
            k: How many similar chunks to return
            
        Returns:
            List of (document_chunk, similarity_score) pairs
        """
        if not self.is_initialized():
            print("No documents in vector store yet!")
            return []
        
        # Convert the query to a vector using the same model
        print(f"Searching for: '{query}'")
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search in our FAISS index for the most similar vectors
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Convert results back to documents with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):  # Make sure index is valid
                document = self.documents[idx]
                similarity_score = float(score)
                results.append((document, similarity_score))
                print(f"Found match (score: {similarity_score:.3f}): {document['content'][:100]}...")
        
        return results
    
    async def delete_document(self, doc_id: str):
        """
        Delete all chunks belonging to a document
        
        Note: FAISS doesn't support deleting individual vectors easily,
        so we mark them as deleted in metadata instead.
        In production, you'd rebuild the index periodically.
        """
        if doc_id not in self.doc_to_chunks:
            print(f"Document {doc_id} not found!")
            return
        
        # Mark all chunks from this document as deleted
        chunk_indices = self.doc_to_chunks[doc_id]
        deleted_count = 0
        
        for idx in chunk_indices:
            if idx < len(self.documents):
                self.documents[idx]["metadata"]["deleted"] = True
                deleted_count += 1
        
        # Remove from our tracking dictionary
        del self.doc_to_chunks[doc_id]
        print(f"Marked {deleted_count} chunks as deleted for document {doc_id}")
    
    def save_to_disk(self, filepath: str):
        """Save the vector store to disk for persistence"""
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save documents and metadata
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'doc_to_chunks': self.doc_to_chunks
            }, f)
        
        print(f"Vector store saved to {filepath}")
    
    def load_from_disk(self, filepath: str):
        """Load the vector store from disk"""
        if os.path.exists(f"{filepath}.faiss") and os.path.exists(f"{filepath}.pkl"):
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load documents and metadata
            with open(f"{filepath}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.doc_to_chunks = data['doc_to_chunks']
            
            print(f"Vector store loaded from {filepath}")
            return True
        return False

# Example of how this works:
if __name__ == "__main__":
    # Create a vector store
    vs = VectorStore()
    
    # Example chunks (normally these come from DocumentProcessor)
    sample_chunks = [
        {
            "content": "The quick brown fox jumps over the lazy dog.",
            "metadata": {"source": "sample.txt", "chunk_id": 0}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "sample.txt", "chunk_id": 1}
        }
    ]
    
    # Add documents (this would be async in real usage)
    import asyncio
    asyncio.run(vs.add_documents(sample_chunks, "doc1"))
    
    # Search for similar content
    results = asyncio.run(vs.similarity_search("What is AI?", k=2))
    
    for doc, score in results:
        print(f"Score: {score:.3f} - Content: {doc['content']}")
