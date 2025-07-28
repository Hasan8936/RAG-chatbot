# backend/document_processor.py

"""
This file handles converting different document types (PDF, DOCX, TXT) 
into text chunks that our AI can understand and search through.

Think of it like a librarian who:
1. Takes books in different formats
2. Reads them and extracts the text
3. Breaks them into manageable chapters (chunks)
4. Labels each chapter with information about where it came from
"""

import fitz  # For reading PDF files
import docx            # For reading Word documents
import os             # For file operations
from typing import List, Dict, Any  # For type hints
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Smart text splitting

class DocumentProcessor:
    """
    This class is like a smart document reader that can:
    - Read different file types
    - Break long documents into smaller, searchable pieces
    - Remember where each piece came from
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize our document processor
        
        Args:
            chunk_size: How many characters per chunk (like paragraph size)
            chunk_overlap: How many characters to overlap between chunks 
                          (this helps maintain context between pieces)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # This is our smart text splitter - it tries to break text at natural points
        # like sentences or paragraphs, not in the middle of words
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,  # How to measure text length
        )
    
    def process_document(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """
        Main function that processes any document type
        
        Args:
            file_path: Where the file is stored on disk
            filename: Original name of the file
            
        Returns:
            List of chunks, each with content and metadata
        """
        # Figure out what type of file this is by looking at the extension
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Call the appropriate extraction method based on file type
        if file_ext == '.pdf':
            text = self._extract_pdf_text(file_path)
        elif file_ext == '.docx':
            text = self._extract_docx_text(file_path)
        elif file_ext == '.txt':
            text = self._extract_txt_text(file_path)
        else:
            raise ValueError(f"Sorry, I don't know how to read {file_ext} files yet!")
        
        # Break the long text into smaller, manageable chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create a list of chunk objects with metadata
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            chunk_docs.append({
                "content": chunk,  # The actual text content
                "metadata": {
                    "source": filename,           # Which file this came from
                    "chunk_id": i,               # Which chunk number this is
                    "total_chunks": len(chunks)  # How many chunks total in this document
                }
            })
        
        return chunk_docs
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """
        Extract text from a PDF file
        
        This is like having a robot read through every page of a PDF
        and type out all the text it sees
        """
        doc = fitz.open(file_path)  # Open the PDF
        text = ""
        
        # Go through each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)  # Load this page
            text += page.get_text()         # Extract text from this page
        
        doc.close()  # Close the PDF when done
        return text
    
    def _extract_docx_text(self, file_path: str) -> str:
        """
        Extract text from a Word document (.docx)
        
        This reads through all paragraphs in a Word document
        """
        doc = docx.Document(file_path)  # Open the Word document
        text = ""
        
        # Go through each paragraph
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"  # Add paragraph text plus a new line
        
        return text
    
    def _extract_txt_text(self, file_path: str) -> str:
        """
        Extract text from a plain text file (.txt)
        
        This is the simplest - just read the file directly
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

# Example of how this class works:
if __name__ == "__main__":
    # Create a document processor
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    
    # Process a document (this would normally be called from main.py)
    # chunks = processor.process_document("sample.pdf", "sample.pdf")
    # print(f"Created {len(chunks)} chunks from the document")
