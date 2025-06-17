"""
Embedding generation using Google Gemini
Implements hierarchical embeddings for document/section/chunk levels
"""
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import google.generativeai as genai
from ..config import settings
from .document_ingestion import DocumentChunk, DocumentMetadata
from loguru import logger

@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    embedding: List[float]
    dimension: int
    model_used: str
    created_at: str

@dataclass
class HierarchicalEmbedding:
    """Hierarchical embedding structure"""
    document_embedding: EmbeddingResult
    section_embeddings: List[EmbeddingResult]
    chunk_embeddings: List[EmbeddingResult]
    metadata: Dict[str, Any]

class GeminiEmbeddingGenerator:
    """Google Gemini embedding generation with hierarchical support"""
    
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=settings.gemini_api_key)
        self.embedding_model = settings.gemini_embedding_model
        
        # Initialize rate limiting
        self._rate_limiter = asyncio.Semaphore(settings.max_concurrent_requests)
        
    async def generate_hierarchical_embeddings(
        self, 
        chunks: List[DocumentChunk], 
        metadata: DocumentMetadata
    ) -> HierarchicalEmbedding:
        """
        Generate hierarchical embeddings for document, sections, and chunks
        
        Args:
            chunks: List of document chunks
            metadata: Document metadata
            
        Returns:
            HierarchicalEmbedding object with all levels
        """
        try:
            # Combine all chunks for document-level embedding
            full_document_text = self._combine_chunks_for_document_embedding(chunks)
            
            # Generate document-level embedding
            document_embedding = await self._generate_single_embedding(
                text=full_document_text,
                embedding_type="document"
            )
            
            # Generate section-level embeddings (group chunks by sections)
            sections = self._group_chunks_into_sections(chunks)
            section_embeddings = []
            
            for section_text in sections:
                section_embedding = await self._generate_single_embedding(
                    text=section_text,
                    embedding_type="section"
                )
                section_embeddings.append(section_embedding)
            
            # Generate chunk-level embeddings
            chunk_embeddings = []
            chunk_tasks = []
            
            # Process chunks in batches to respect rate limits
            batch_size = min(settings.max_concurrent_requests, len(chunks))
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_tasks = [
                    self._generate_single_embedding(chunk.content, "chunk") 
                    for chunk in batch
                ]
                batch_results = await asyncio.gather(*batch_tasks)
                chunk_embeddings.extend(batch_results)
                
                # Small delay between batches to respect rate limits
                if i + batch_size < len(chunks):
                    await asyncio.sleep(0.1)
              # Create hierarchical embedding structure
            from datetime import datetime
            hierarchical_metadata = {
                "document_hash": metadata.document_hash,
                "source_path": metadata.source_path,
                "total_chunks": len(chunks),
                "total_sections": len(sections),
                "embedding_model": self.embedding_model,
                "generation_timestamp": datetime.now().isoformat()
            }
            
            result = HierarchicalEmbedding(
                document_embedding=document_embedding,
                section_embeddings=section_embeddings,
                chunk_embeddings=chunk_embeddings,
                metadata=hierarchical_metadata
            )
            
            logger.info(f"Generated hierarchical embeddings for {metadata.source_path}: "
                       f"{len(chunk_embeddings)} chunks, {len(section_embeddings)} sections")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating hierarchical embeddings: {str(e)}")
            raise
    
    async def _generate_single_embedding(self, text: str, embedding_type: str) -> EmbeddingResult:
        """Generate a single embedding with rate limiting"""
        async with self._rate_limiter:
            try:
                # Clean and prepare text
                cleaned_text = self._clean_text_for_embedding(text)
                  # Generate embedding using Gemini
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=cleaned_text,
                    task_type="retrieval_document"
                )
                
                embedding_vector = result['embedding']
                
                from datetime import datetime
                return EmbeddingResult(
                    embedding=embedding_vector,
                    dimension=len(embedding_vector),
                    model_used=self.embedding_model,
                    created_at=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Error generating {embedding_type} embedding: {str(e)}")                # Return zero embedding as fallback
                from datetime import datetime
                return EmbeddingResult(
                    embedding=[0.0] * 768,  # Default dimension
                    dimension=768,
                    model_used=self.embedding_model,
                    created_at=datetime.now().isoformat()
                )
    
    def _clean_text_for_embedding(self, text: str) -> str:
        """Clean and prepare text for embedding generation"""
        # Remove page markers and excessive whitespace
        import re
        
        # Remove page markers
        text = re.sub(r'\[PAGE \d+\]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Trim to reasonable length for embedding
        max_length = 8000  # Conservative limit for Gemini
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text.strip()
    
    def _combine_chunks_for_document_embedding(self, chunks: List[DocumentChunk]) -> str:
        """Combine chunks intelligently for document-level embedding"""
        # Strategy: Use first few chunks, last few chunks, and sample from middle
        # to create a representative document embedding
        
        if len(chunks) <= 5:
            return " ".join([chunk.content for chunk in chunks])
        
        # Take first 2 chunks
        combined_text = chunks[0].content + " " + chunks[1].content
        
        # Sample from middle chunks
        middle_start = len(chunks) // 3
        middle_end = 2 * len(chunks) // 3
        middle_chunks = chunks[middle_start:middle_end:max(1, (middle_end - middle_start) // 3)]
        
        for chunk in middle_chunks:
            combined_text += " " + chunk.content
        
        # Take last 2 chunks
        combined_text += " " + chunks[-2].content + " " + chunks[-1].content
        
        return combined_text
      def _group_chunks_into_sections(self, chunks: List[DocumentChunk]) -> List[str]:
        """Group chunks into logical sections for section-level embeddings"""
        sections = []
        
        # Simple strategy: group consecutive chunks into sections
        # More sophisticated strategies could use topic modeling or structure detection
        
        section_size = max(3, len(chunks) // 5)  # Aim for ~5 sections
        
        for i in range(0, len(chunks), section_size):
            section_chunks = chunks[i:i + section_size]
            section_text = " ".join([chunk.content for chunk in section_chunks])
            sections.append(section_text)
        
        return sections
    
    async def generate_query_embedding(self, query: str) -> EmbeddingResult:
        """Generate embedding for a query"""
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query"
            )
            
            embedding_vector = result['embedding']
            
            from datetime import datetime
            return EmbeddingResult(
                embedding=embedding_vector,
                dimension=len(embedding_vector),
                model_used=self.embedding_model,
                created_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

# Export the main embedding generator
embedding_generator = GeminiEmbeddingGenerator()
