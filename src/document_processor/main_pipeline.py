"""
Main document processor that orchestrates the complete Phase 1 pipeline
Handles multi-format document ingestion, embedding generation, and vector storage
"""
import asyncio
from typing import Dict, Any, List
from pathlib import Path
import time

from .document_ingestion import document_processor, DocumentChunk, DocumentMetadata
from .embedding_generator import embedding_generator, HierarchicalEmbedding
from .vector_database import vector_db
from ..config import settings
from loguru import logger

class DocumentProcessingPipeline:
    """Complete document processing pipeline for Phase 1"""
    
    def __init__(self):
        self.processor = document_processor
        self.embedder = embedding_generator
        self.vector_store = vector_db
        self.processing_stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "total_processing_time": 0.0
        }
    
    async def process_single_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single document through the complete pipeline
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Processing result with statistics
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting document processing pipeline for: {file_path}")
            
            # Step 1: Document Ingestion and Chunking
            logger.info("Step 1: Document ingestion and intelligent chunking...")
            document_result = await self.processor.process_document(file_path)
            
            chunks = document_result["chunks"]
            metadata = document_result["metadata"]
            
            logger.info(f"Created {len(chunks)} intelligent chunks")
            
            # Step 2: Generate Hierarchical Embeddings
            logger.info("Step 2: Generating hierarchical embeddings...")
            hierarchical_embeddings = await self.embedder.generate_hierarchical_embeddings(
                chunks, metadata
            )
            
            logger.info(f"Generated embeddings: {len(hierarchical_embeddings.chunk_embeddings)} chunks, "
                       f"{len(hierarchical_embeddings.section_embeddings)} sections")
            
            # Step 3: Store in Vector Database
            logger.info("Step 3: Storing in vector database...")
            storage_result = await self.vector_store.store_document_embeddings(
                chunks, hierarchical_embeddings
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update statistics
            self.processing_stats["documents_processed"] += 1
            self.processing_stats["chunks_created"] += len(chunks)
            self.processing_stats["embeddings_generated"] += len(hierarchical_embeddings.chunk_embeddings)
            self.processing_stats["total_processing_time"] += processing_time
            
            # Prepare comprehensive result
            result = {
                "success": True,
                "file_path": file_path,
                "document_metadata": {
                    "source_path": metadata.source_path,
                    "file_type": metadata.file_type,
                    "file_size": metadata.file_size,
                    "document_hash": metadata.document_hash,
                    "total_words": metadata.total_words,
                    "total_pages": metadata.total_pages
                },
                "processing_results": {
                    "chunks_created": len(chunks),
                    "chunk_embeddings": len(hierarchical_embeddings.chunk_embeddings),
                    "section_embeddings": len(hierarchical_embeddings.section_embeddings),
                    "document_embedding": 1,
                    "processing_time_seconds": round(processing_time, 2)
                },
                "storage_results": storage_result,
                "quality_metrics": {
                    "avg_chunk_size": sum(len(chunk.content) for chunk in chunks) // len(chunks),
                    "embedding_dimension": hierarchical_embeddings.chunk_embeddings[0].dimension if hierarchical_embeddings.chunk_embeddings else 0,
                    "chunks_with_page_numbers": sum(1 for chunk in chunks if chunk.page_number is not None)
                }
            }
            
            logger.info(f"Successfully processed {file_path} in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_result = {
                "success": False,
                "file_path": file_path,
                "error": str(e),
                "processing_time_seconds": round(processing_time, 2)
            }
            
            logger.error(f"Failed to process {file_path}: {str(e)}")
            return error_result
    
    async def process_multiple_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Process multiple documents concurrently
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Batch processing results
        """
        start_time = time.time()
        
        logger.info(f"Starting batch processing for {len(file_paths)} documents")
        
        # Process documents with controlled concurrency
        semaphore = asyncio.Semaphore(3)  # Limit concurrent processing
        
        async def process_with_semaphore(file_path):
            async with semaphore:
                return await self.process_single_document(file_path)
        
        # Execute all processing tasks
        tasks = [process_with_semaphore(file_path) for file_path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append({"error": str(result)})
            elif result.get("success", False):
                successful_results.append(result)
            else:
                failed_results.append(result)
        
        total_time = time.time() - start_time
        
        batch_result = {
            "batch_summary": {
                "total_documents": len(file_paths),
                "successful_documents": len(successful_results),
                "failed_documents": len(failed_results),
                "total_processing_time": round(total_time, 2),
                "avg_time_per_document": round(total_time / len(file_paths), 2) if file_paths else 0
            },
            "successful_results": successful_results,
            "failed_results": failed_results,
            "pipeline_stats": self.processing_stats.copy()
        }
        
        logger.info(f"Batch processing completed: {len(successful_results)}/{len(file_paths)} successful")
        return batch_result
    
    async def process_directory(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
        """
        Process all supported documents in a directory
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search subdirectories
            
        Returns:
            Directory processing results
        """
        try:
            directory = Path(directory_path)
            if not directory.exists():
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            
            # Find all supported files
            supported_extensions = {'.pdf', '.docx', '.txt', '.html', '.htm', '.md', '.markdown'}
            file_paths = []
            
            if recursive:
                for ext in supported_extensions:
                    file_paths.extend(directory.rglob(f"*{ext}"))
            else:
                for ext in supported_extensions:
                    file_paths.extend(directory.glob(f"*{ext}"))
            
            # Convert to string paths
            file_paths = [str(path) for path in file_paths]
            
            logger.info(f"Found {len(file_paths)} supported documents in {directory_path}")
            
            if not file_paths:
                return {
                    "directory_path": directory_path,
                    "files_found": 0,
                    "message": "No supported documents found"
                }
            
            # Process all found documents
            result = await self.process_multiple_documents(file_paths)
            result["directory_path"] = directory_path
            result["files_found"] = len(file_paths)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            return {
                "directory_path": directory_path,
                "error": str(e),
                "success": False
            }
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            "processing_stats": self.processing_stats,
            "vector_database_stats": vector_stats,
            "configuration": {
                "max_chunk_size": settings.max_chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "embedding_model": settings.gemini_embedding_model,
                "vector_collection": settings.chroma_collection_name
            },
            "performance_metrics": {
                "avg_processing_time": (
                    self.processing_stats["total_processing_time"] / 
                    max(1, self.processing_stats["documents_processed"])
                ),
                "avg_chunks_per_document": (
                    self.processing_stats["chunks_created"] / 
                    max(1, self.processing_stats["documents_processed"])
                )
            }
        }
    
    async def search_documents(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search processed documents using the query
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Search results
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedder.generate_query_embedding(query)
            
            # Perform hybrid search
            results = await self.vector_store.hybrid_search(
                query_embedding.embedding,
                query,
                n_results
            )
            
            return {
                "query": query,
                "results_count": len(results),
                "results": results,
                "search_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "results": []
            }

# Export the main pipeline
document_pipeline = DocumentProcessingPipeline()
