# backend/rag_pipeline.py

"""
This is the "brain" of our RAG system - the RAG Pipeline.

RAG stands for "Retrieval-Augmented Generation":
- RETRIEVAL: Find relevant documents from our vector store
- AUGMENTED: Add this information to our prompt
- GENERATION: Use AI (like GPT) to generate an answer based on the retrieved info

Think of it like a smart research assistant that:
1. Searches through your documents for relevant information
2. Reads the relevant parts
3. Writes a comprehensive answer based on what it found
4. Tells you exactly which documents it used
"""

import openai  # For calling OpenAI's GPT models
from typing import List, Dict, Any  # For type hints
import os      # For environment variables

class RAGPipeline:
    """
    This class orchestrates the entire RAG process:
    1. Takes a user question
    2. Finds relevant document chunks
    3. Builds a context-rich prompt
    4. Asks GPT to generate an answer
    5. Returns the answer with source citations
    """
    
    def __init__(self, vector_store):
        """
        Initialize the RAG pipeline
        
        Args:
            vector_store: Our VectorStore instance for searching documents
        """
        self.vector_store = vector_store
        
        # Set up OpenAI API key
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables!")
    
    async def query(self, question: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process a user question through the complete RAG pipeline
        
        This is the main function that does all the RAG magic!
        
        Args:
            question: The user's question
            chat_history: Previous conversation for context
            
        Returns:
            Dictionary with answer, sources, and confidence score
        """
        print(f"\n🔍 Processing query: '{question}'")
        
        # STEP 1: RETRIEVAL - Find relevant documents
        print("📚 Step 1: Searching for relevant documents...")
        search_results = await self.vector_store.similarity_search(question, k=5)
        
        # Check if we found any documents
        if not search_results:
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents to answer your question. Could you try rephrasing your question or upload more documents?",
                "sources": [],
                "confidence": 0.0
            }
        
        # STEP 2: Filter out deleted documents
        print("🧹 Step 2: Filtering valid documents...")
        valid_results = [
            (doc, score) for doc, score in search_results 
            if not doc["metadata"].get("deleted", False)
        ]
        
        if not valid_results:
            return {
                "answer": "I found some documents, but they seem to have been deleted. Please upload some documents first!",
                "sources": [],
                "confidence": 0.0
            }
        
        # STEP 3: Build context from retrieved documents
        print("📝 Step 3: Building context from retrieved documents...")
        context_parts = []
        sources = []
        
        for i, (doc, score) in enumerate(valid_results):
            # Create a formatted context entry
            context_entry = f"[Source {i+1}: {doc['metadata']['source']}]\n{doc['content']}"
            context_parts.append(context_entry)
            
            # Store source information for citation
            sources.append({
                "source": doc["metadata"]["source"],
                "chunk_id": doc["metadata"]["chunk_id"],
                "confidence": float(score),
                "preview": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
            })
        
        # Join all context parts
        context = "\n\n" + "="*50 + "\n\n".join(context_parts)
        
        # STEP 4: Build chat history context (if provided)
        print("💬 Step 4: Adding chat history context...")
        history_context = ""
        if chat_history:
            recent_history = chat_history[-4:]  # Last 4 messages for context
            history_parts = []
            
            for msg in recent_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_parts.append(f"{role.title()}: {content}")
            
            if history_parts:
                history_context = f"\n\nPrevious conversation:\n" + "\n".join(history_parts)
        
        # STEP 5: AUGMENTED GENERATION - Create the prompt
        print("🤖 Step 5: Building AI prompt...")
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(question, context, history_context)
        
        # STEP 6: Generate response using OpenAI
        print("✨ Step 6: Generating AI response...")
        try:
            response = await self._call_openai(system_prompt, user_prompt)
            
            # Calculate average confidence from search results
            avg_confidence = sum(score for _, score in valid_results) / len(valid_results)
            
            print(f"✅ Successfully generated response with {len(sources)} sources")
            
            return {
                "answer": response,
                "sources": sources,
                "confidence": float(avg_confidence)
            }
            
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            return {
                "answer": f"I found relevant documents but encountered an error while generating the response: {str(e)}",
                "sources": sources,
                "confidence": 0.0
            }
    
    def _create_system_prompt(self) -> str:
        """
        Create the system prompt that tells GPT how to behave
        
        This is like giving instructions to a human assistant about how to do their job
        """
        return """You are a helpful AI assistant that answers questions based ONLY on the provided document context.

Your job is to:
1. Read through the provided context carefully
2. Answer the user's question using ONLY information from the context
3. Be accurate and comprehensive
4. If the context doesn't contain enough information, say so honestly
5. Synthesize information from multiple sources when relevant
6. Maintain a helpful and professional tone

Important rules:
- NEVER make up information that's not in the context
- If you're unsure, say "Based on the provided documents..."
- If the context is insufficient, suggest what additional information might be needed
- Always be honest about the limitations of your knowledge based on the provided context"""
    
    def _create_user_prompt(self, question: str, context: str, history_context: str) -> str:
        """
        Create the user prompt with the question and all context
        
        This assembles all the information the AI needs to answer the question
        """
        prompt_parts = [
            "Please answer the following question based on the document context provided below."
        ]
        
        # Add chat history if available
        if history_context.strip():
            prompt_parts.append(f"\nFor additional context, here's our recent conversation:{history_context}")
        
        # Add document context
        prompt_parts.append(f"\nDocument context:{context}")
        
        # Add the actual question
        prompt_parts.append(f"\nQuestion: {question}")
        
        # Instructions for the answer
        prompt_parts.append(
            "\nPlease provide a helpful and accurate answer based on the context above. "
            "If you reference specific information, it should come from the provided sources."
        )
        
        return "\n".join(prompt_parts)
    
    async def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """
        Make the actual call to OpenAI's API
        
        This is where we send our carefully crafted prompt to GPT and get back an answer
        """
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # Use gpt-4 if you have access and want better quality
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,        # Low temperature for more consistent, factual responses
                max_tokens=800,         # Maximum length of response
                top_p=0.9              # Focus on most likely responses
            )
            
            return response.choices[0].message.content.strip()
            
        except openai.RateLimitError:
            raise Exception("OpenAI rate limit exceeded. Please wait a moment and try again.")
        except openai.AuthenticationError:
            raise Exception("OpenAI authentication failed. Please check your API key.")
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

# Example of how this works:
if __name__ == "__main__":
    from vector_store import VectorStore
    import asyncio
    
    # This would normally be set up in main.py
    vs = VectorStore()
    pipeline = RAGPipeline(vs)
    
    # Example usage (after documents are loaded)
    async def test_pipeline():
        result = await pipeline.query("What is machine learning?")
        print("Answer:", result["answer"])
        print("Sources:", len(result["sources"]))
        print("Confidence:", result["confidence"])
    
    # asyncio.run(test_pipeline())
    print("RAG Pipeline ready! Use this in main.py")
