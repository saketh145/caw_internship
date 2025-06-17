"""
ChromaDB vector database setup and management
Handles storage of document chunks, embeddings, and metadata
"""

# Try to import ChromaDB, fall back to simple vector store if not available
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    from .simple_vector_store import PersistentClient
    ChromaSettings = None

from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import uuid

from ..config import settings
from .document_ingestion import DocumentChunk
from .embedding_generator import HierarchicalEmbedding, EmbeddingResult
from loguru import logger

class VectorDatabase:
    """ChromaDB vector database manager for the Q&A system"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self._initialize_database()    
    def _initialize_database(self):
        """Initialize ChromaDB client and collection"""
        try:
            if CHROMADB_AVAILABLE:
                # Initialize ChromaDB client with persistence
                self.client = chromadb.PersistentClient(
                    path=settings.chroma_persist_directory,
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                logger.info("Using ChromaDB for vector storage")
            else:
                # Fallback to simple vector store
                self.client = PersistentClient(path=settings.chroma_persist_directory)
                logger.warning("ChromaDB not available, using simple in-memory vector store")
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=settings.chroma_collection_name,
                metadata={
                    "description": "Intelligent Document Q&A System with Memory",
                    "created_at": datetime.now().isoformat(),
                    "embedding_model": settings.gemini_embedding_model
                }
            )
            
            logger.info(f"Initialized ChromaDB collection: {settings.chroma_collection_name}")
            logger.info(f"Collection count: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Error initializing vector database: {str(e)}")
            raise
    
    async def store_document_embeddings(
        self, 
        chunks: List[DocumentChunk], 
        hierarchical_embeddings: HierarchicalEmbedding
    ) -> Dict[str, Any]:
        """
        Store document chunks and their embeddings in the vector database
        
        Args:
            chunks: List of document chunks
            hierarchical_embeddings: Generated hierarchical embeddings
            
        Returns:
            Storage result summary
        """
        try:
            stored_ids = []
            stored_count = 0
            
            # Prepare data for batch insertion
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for i, chunk in enumerate(chunks):
                chunk_embedding = hierarchical_embeddings.chunk_embeddings[i]
                
                # Create unique ID for this chunk
                chunk_id = f"{chunk.chunk_id}_{uuid.uuid4().hex[:8]}"
                  # Prepare metadata with comprehensive information
                metadata = {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.metadata["source_file"],  # Use 'source' for Q&A retrieval
                    "source_file": chunk.metadata["source_file"],  # Keep original for compatibility
                    "file_type": chunk.metadata["file_type"],
                    "chunk_index": chunk.chunk_index,
                    "document_hash": chunk.metadata["document_hash"],
                    "word_count": chunk.metadata["word_count"],
                    "char_count": chunk.metadata["char_count"],
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "created_at": datetime.now().isoformat(),
                    "embedding_model": chunk_embedding.model_used,
                    "embedding_dimension": chunk_embedding.dimension
                }
                
                # Add page number if available
                if chunk.page_number:
                    metadata["page_number"] = chunk.page_number
                
                # Add hierarchical context
                metadata["has_document_embedding"] = True
                metadata["has_section_embedding"] = i < len(hierarchical_embeddings.section_embeddings)
                
                ids.append(chunk_id)
                embeddings.append(chunk_embedding.embedding)
                metadatas.append(metadata)
                documents.append(chunk.content)
                stored_ids.append(chunk_id)
            
            # Batch insert into ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            stored_count = len(ids)
            
            # Store document-level metadata separately
            await self._store_document_metadata(hierarchical_embeddings)
            
            logger.info(f"Stored {stored_count} chunks in vector database")
            
            return {
                "stored_count": stored_count,
                "stored_ids": stored_ids,
                "collection_name": settings.chroma_collection_name,
                "total_collection_size": self.collection.count()
            }
            
        except Exception as e:
            logger.error(f"Error storing document embeddings: {str(e)}")
            raise
    
    async def _store_document_metadata(self, hierarchical_embeddings: HierarchicalEmbedding):
        """Store document-level and section-level embeddings as metadata"""
        try:
            # Store document-level embedding
            doc_metadata_id = f"doc_meta_{hierarchical_embeddings.metadata['document_hash']}"
            
            doc_metadata = {
                "type": "document_metadata",
                "document_hash": hierarchical_embeddings.metadata["document_hash"],
                "source_path": hierarchical_embeddings.metadata["source_path"],
                "total_chunks": hierarchical_embeddings.metadata["total_chunks"],
                "total_sections": hierarchical_embeddings.metadata["total_sections"],
                "embedding_model": hierarchical_embeddings.metadata["embedding_model"],
                "created_at": datetime.now().isoformat()
            }
            
            # Store document embedding
            self.collection.add(
                ids=[doc_metadata_id],
                embeddings=[hierarchical_embeddings.document_embedding.embedding],
                metadatas=[doc_metadata],
                documents=[f"Document metadata for {hierarchical_embeddings.metadata['source_path']}"]
            )
            
            # Store section embeddings
            for i, section_embedding in enumerate(hierarchical_embeddings.section_embeddings):
                section_id = f"section_{hierarchical_embeddings.metadata['document_hash']}_{i}"
                
                section_metadata = {
                    "type": "section_metadata",
                    "document_hash": hierarchical_embeddings.metadata["document_hash"],
                    "section_index": i,
                    "source_path": hierarchical_embeddings.metadata["source_path"],
                    "embedding_model": section_embedding.model_used,
                    "created_at": datetime.now().isoformat()
                }
                
                self.collection.add(
                    ids=[section_id],
                    embeddings=[section_embedding.embedding],
                    metadatas=[section_metadata],
                    documents=[f"Section {i} metadata"]
                )
            
        except Exception as e:
            logger.error(f"Error storing document metadata: {str(e)}")
    
    async def similarity_search(
        self, 
        query_embedding: List[float], 
        n_results: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search in the vector database
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_criteria: Optional metadata filters
            
        Returns:
            List of similar chunks with metadata
        """
        try:            # Prepare where clause for filtering - only get content chunks, not metadata
            where_clause = None
            if filter_criteria:
                where_clause = filter_criteria
            # Note: We'll filter out metadata entries in post-processing since 
            # the simple vector store may not support complex where clauses
              # Perform similarity search
            if where_clause:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where_clause,
                    include=["documents", "metadatas", "distances"]
                )
            else:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
            
            # Format results and filter out metadata entries
            formatted_results = []
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                
                # Skip metadata entries (those with type field)
                if metadata.get('type') in ['document_metadata', 'section_metadata']:
                    continue
                
                result = {
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": metadata,
                    "similarity_score": 1.0 - results['distances'][0][i],  # Convert distance to similarity
                    "distance": results['distances'][0][i]
                }
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar chunks")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return []
    
    async def hybrid_search(
        self, 
        query_embedding: List[float],
        query_text: str,
        n_results: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword matching
        
        Args:
            query_embedding: Query embedding for semantic search
            query_text: Query text for keyword search
            n_results: Number of results to return
            semantic_weight: Weight for semantic similarity
            keyword_weight: Weight for keyword matching
            
        Returns:
            List of results with combined scores
        """
        try:
            # Get semantic search results
            semantic_results = await self.similarity_search(query_embedding, n_results * 2)
            
            # Get all documents for keyword search
            all_docs = self.collection.get(
                include=["documents", "metadatas"],
                where={"type": {"$ne": "document_metadata"}}
            )
            
            # Perform keyword search using simple text matching
            keyword_scores = {}
            query_terms = query_text.lower().split()
            
            for i, doc in enumerate(all_docs['documents']):
                doc_lower = doc.lower()
                score = 0
                for term in query_terms:
                    score += doc_lower.count(term)
                
                # Normalize by document length
                if len(doc) > 0:
                    keyword_scores[all_docs['ids'][i]] = score / len(doc.split())
            
            # Combine semantic and keyword scores
            combined_results = []
            for result in semantic_results:
                doc_id = result['id']
                semantic_score = result['similarity_score']
                keyword_score = keyword_scores.get(doc_id, 0)
                
                # Calculate combined score
                combined_score = (semantic_weight * semantic_score + 
                                keyword_weight * keyword_score)
                
                result['combined_score'] = combined_score
                result['semantic_score'] = semantic_score
                result['keyword_score'] = keyword_score
                combined_results.append(result)
            
            # Sort by combined score and return top results
            combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            logger.info(f"Hybrid search returned {len(combined_results[:n_results])} results")
            return combined_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {str(e)}")
            return await self.similarity_search(query_embedding, n_results)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database collection"""
        try:
            count = self.collection.count()
            
            # Get sample of metadata to analyze
            sample_size = min(100, count)
            if count > 0:
                sample = self.collection.get(limit=sample_size, include=["metadatas"])
                
                # Analyze file types
                file_types = {}
                total_chunks = 0
                
                for metadata in sample['metadatas']:
                    if metadata.get('type') != 'document_metadata':
                        file_type = metadata.get('file_type', 'unknown')
                        file_types[file_type] = file_types.get(file_type, 0) + 1
                        total_chunks += 1
                
                return {
                    "total_documents": count,
                    "sample_size": sample_size,
                    "file_types": file_types,
                    "collection_name": settings.chroma_collection_name,
                    "embedding_model": settings.gemini_embedding_model
                }
            else:
                return {
                    "total_documents": 0,
                    "collection_name": settings.chroma_collection_name,
                    "embedding_model": settings.gemini_embedding_model
                }
                
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}
    
    def delete_document(self, document_hash: str) -> bool:
        """Delete all chunks for a specific document"""
        try:
            # Find all chunks for this document
            results = self.collection.get(
                where={"document_hash": document_hash},
                include=["ids"]
            )
            
            if results['ids']:
                # Delete all chunks
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_hash}")
                return True
            else:
                logger.warning(f"No chunks found for document {document_hash}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document {document_hash}: {str(e)}")
            return False
    
    def update_document_metadata(self, document_id: str, new_metadata: Dict[str, Any]) -> bool:
        """Update metadata for a specific document chunk"""
        try:
            # Get the current document
            result = self.collection.get(
                ids=[document_id],
                include=["metadatas"]
            )
            
            if not result['ids']:
                logger.warning(f"Document {document_id} not found")
                return False
            
            # Update the metadata
            current_metadata = result['metadatas'][0]
            current_metadata.update(new_metadata)
            
            # Update in ChromaDB
            self.collection.update(
                ids=[document_id],
                metadatas=[current_metadata]
            )
            
            logger.info(f"Updated metadata for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating metadata for document {document_id}: {str(e)}")
            return False

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from the collection"""
        try:
            # Get all documents
            results = self.collection.get(
                include=["ids", "metadatas", "documents"]
            )
            
            documents = []
            for i, doc_id in enumerate(results['ids']):
                documents.append({
                    'id': doc_id,
                    'metadata': results['metadatas'][i],
                    'content': results['documents'][i]
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting all documents: {str(e)}")
            return []

# Export the main vector database instance
vector_db = VectorDatabase()
