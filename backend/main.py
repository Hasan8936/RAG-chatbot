# backend/main.py

"""
This is the main FastAPI application - the "server" that handles web requests.

Think of this as the front desk of a hotel:
- Guests (frontend/users) make requests
- The front desk (FastAPI) receives these requests
- It delegates tasks to appropriate departments (our other modules)
- It sends responses back to the guests

FastAPI automatically creates a web API with documentation at http://localhost:8000/docs
"""

from fastapi import FastAPI, UploadFile, File, HTTPException  # Web framework
from fastapi.middleware.cors import CORSMiddleware              # Allow frontend to connect
from pydantic import BaseModel                                 # Data validation
import os              # File operations
import tempfile        # Temporary files
import uuid           # Unique IDs
from typing import List, Dict, Any  # Type hints
import asyncio        # Async operations
from dotenv import load_dotenv  # Load environment variables

# Import our custom modules (the ones we just created!)
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

# Load environment variables from .env file
load_dotenv()

# Create FastAPI application
app = FastAPI(
    title="RAG Document Q&A System",
    description="Upload documents and ask questions about them using AI",
    version="1.0.0"
)

# Configure CORS (Cross-Origin Resource Sharing)
# This allows our React frontend to talk to our Python backend
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Use environment variable for origins
    allow_credentials=True,
    allow_methods=["*"],      # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],      # Allow all headers
)

# Initialize our components (like setting up departments in our hotel)
print("🚀 Initializing RAG system components...")
document_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
vector_store = VectorStore()
rag_pipeline = RAGPipeline(vector_store)
print("✅ Components initialized successfully!")

# Pydantic models for request/response validation
# These define the "shape" of data that comes in and goes out

class QueryRequest(BaseModel):
    """What a user sends when asking a question"""
    question: str
    chat_history: List[Dict[str, str]] = []  # Optional chat history

class QueryResponse(BaseModel):
    """What we send back after answering a question"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float

class DocumentInfo(BaseModel):
    """Information about an uploaded document"""
    id: str
    filename: str
    chunks_count: int
    status: str

# In-memory storage for document metadata
# In production, this would be a proper database
documents_db: Dict[str, DocumentInfo] = {}

# API ENDPOINTS (like different services at our hotel front desk)

@app.get("/")
def read_root():
    """
    Root endpoint - like the hotel's welcome sign
    This is what you see when you visit http://localhost:8000
    """
    return {
        "message": "Welcome to the RAG Document Q&A System API!", 
        "status": "running",
        "docs": "Visit /docs for API documentation"
    }

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document
    
    This is like a guest checking in - we:
    1. Receive their document
    2. Process it (read and chunk it)
    3. Store it in our searchable database
    4. Give them a receipt (document ID)
    """
    try:
        print(f"📤 Received upload request for: {file.filename}")
        
        # STEP 1: Validate file type
        allowed_types = ['.pdf', '.docx', '.txt']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type '{file_ext}'. Allowed types: {allowed_types}"
            )
        
        print(f"✅ File type '{file_ext}' is supported")
        
        # STEP 2: Create unique document ID and temporary file
        doc_id = str(uuid.uuid4())  # Generate unique ID like "abc123-def456-..."
        print(f"📝 Generated document ID: {doc_id}")
        
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()  # Read the uploaded file
            temp_file.write(content)     # Write it to temp file
            temp_path = temp_file.name   # Remember where we put it
        
        print(f"💾 Saved uploaded file to temporary location: {temp_path}")
        
        try:
            # STEP 3: Process the document (extract text and create chunks)
            print("🔄 Processing document...")
            chunks = document_processor.process_document(temp_path, file.filename)
            print(f"✅ Created {len(chunks)} chunks from document")
            
            # STEP 4: Store chunks in vector database
            print("🗃️ Adding chunks to vector store...")
            await vector_store.add_documents(chunks, doc_id)
            print("✅ Chunks stored in vector database")
            
            # STEP 5: Save document metadata
            documents_db[doc_id] = DocumentInfo(
                id=doc_id,
                filename=file.filename,
                chunks_count=len(chunks),
                status="processed"
            )
            print(f"📋 Document metadata saved for {file.filename}")
            
            # Return success response
            return {
                "document_id": doc_id,
                "filename": file.filename,
                "chunks_processed": len(chunks),
                "status": "success",
                "message": f"Successfully processed '{file.filename}' into {len(chunks)} searchable chunks"
            }
            
        finally:
            # STEP 6: Clean up temporary file (like housekeeping)
            try:
                os.unlink(temp_path)
                print(f"🧹 Cleaned up temporary file: {temp_path}")
            except:
                pass  # If cleanup fails, it's not critical
            
    except HTTPException:
        # Re-raise HTTP exceptions (these are expected errors)
        raise
    except Exception as e:
        # Log unexpected errors and return a generic error message
        print(f"❌ Unexpected error processing document: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred while processing the document: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the document database with a question
    
    This is like asking the concierge a question - they:
    1. Search through all available information
    2. Find the most relevant details
    3. Provide a comprehensive answer
    4. Tell you exactly where they found the information
    """
    try:
        print(f"❓ Received query: '{request.question}'")
        
        # Check if any documents have been uploaded
        if not documents_db:
            raise HTTPException(
                status_code=400, 
                detail="No documents have been uploaded yet. Please upload some documents first!"
            )
        
        print(f"📚 Searching through {len(documents_db)} uploaded documents...")
        
        # Process the query through our RAG pipeline
        result = await rag_pipeline.query(request.question, request.chat_history)
        
        print(f"✅ Query processed successfully")
        print(f"📊 Found {len(result['sources'])} relevant sources")
        print(f"🎯 Confidence score: {result['confidence']:.2f}")
        
        # Return the structured response
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"]
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"❌ Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing your question: {str(e)}"
        )

@app.get("/documents")
def list_documents():
    """
    List all uploaded documents
    
    This is like asking for a list of all guests currently checked in
    """
    print(f"📋 Listing {len(documents_db)} uploaded documents")
    return {
        "documents": list(documents_db.values()),
        "total_count": len(documents_db)
    }

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document from the system
    
    This is like a guest checking out - we remove all their information
    """
    try:
        if doc_id not in documents_db:
            raise HTTPException(
                status_code=404, 
                detail=f"Document with ID '{doc_id}' not found"
            )
        
        # Get document info before deletion
        doc_info = documents_db[doc_id]
        print(f"🗑️ Deleting document: {doc_info.filename} (ID: {doc_id})")
        
        # Remove from vector store
        await vector_store.delete_document(doc_id)
        
        # Remove from our metadata database
        del documents_db[doc_id]
        
        print(f"✅ Successfully deleted document: {doc_info.filename}")
        
        return {
            "message": f"Document '{doc_info.filename}' deleted successfully",
            "deleted_document_id": doc_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while deleting the document: {str(e)}"
        )

@app.get("/health")
def health_check():
    """
    Health check endpoint
    
    This is like checking if all hotel systems are working properly
    """
    return {
        "status": "healthy",
        "documents_count": len(documents_db),
        "vector_store_initialized": vector_store.is_initialized(),
        "openai_api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "message": "All systems operational"
    }

@app.get("/stats")
def get_stats():
    """
    Get system statistics
    
    This provides an overview of system usage
    """
    total_chunks = sum(doc.chunks_count for doc in documents_db.values())
    
    return {
        "total_documents": len(documents_db),
        "total_chunks": total_chunks,
        "average_chunks_per_document": total_chunks / len(documents_db) if documents_db else 0,
        "vector_store_size": vector_store.index.ntotal if vector_store.is_initialized() else 0
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors gracefully"""
    return {"error": "Endpoint not found", "message": "The requested endpoint does not exist"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors gracefully"""
    return {"error": "Internal server error", "message": "An unexpected error occurred"}

# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Run this when the server starts
    
    Like opening the hotel for business - we make sure everything is ready
    """
    print("🏨 Starting RAG Document Q&A System...")
    print("🔧 Checking configuration...")
    
    # Check if OpenAI API key is configured
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not found in environment variables!")
        print("   The system will not be able to generate responses until this is configured.")
    else:
        print("✅ OpenAI API key configured")
    
    print("🎉 System startup complete!")
    print("📖 API documentation available at: http://localhost:8000/docs")
    print("🏠 Frontend should connect to: http://localhost:8000")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Run this when the server shuts down
    
    Like closing the hotel - we clean up properly
    """
    print("🛑 Shutting down RAG Document Q&A System...")
    
    # In a production system, you might want to:
    # - Save the vector store to disk
    # - Close database connections
    # - Clean up temporary files
    
    print("👋 Shutdown complete!")

# Development server runner
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting development server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API docs will be available at: http://localhost:8000/docs")
    
    # Run the server
    uvicorn.run(
        app, 
        host="0.0.0.0",    # Accept connections from any IP
        port=8000,         # Port number
        reload=True,       # Auto-reload when code changes (development only)
        log_level="info"   # Logging level
    )