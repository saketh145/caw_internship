"""
Dynamic Document Processing System
Handles real-time document uploads with automatic chunking and embedding storage
"""

import asyncio
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import hashlib
from datetime import datetime
import uuid
from loguru import logger

from .document_ingestion import document_processor, DocumentChunk
from .embedding_generator import embedding_generator
from .vector_database import vector_db
from ..config import settings

class DynamicDocumentProcessor:
    """Enhanced document processor for real-time dynamic handling"""
    
    def __init__(self):
        self.processor = document_processor
        self.embedder = embedding_generator
        self.vector_store = vector_db
        self.upload_dir = Path("uploads")
        self.processed_dir = Path("processed_documents")
        self.upload_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        # Document registry for tracking
        self.document_registry = {}
        
        # Processing queue for background processing
        self.processing_queue = asyncio.Queue()
        self.background_task = None
        
    async def start_background_processor(self):
        """Start the background document processing task"""
        if self.background_task is None:
            self.background_task = asyncio.create_task(self._background_processor())
            logger.info("Background document processor started")
    
    async def stop_background_processor(self):
        """Stop the background document processing task"""
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
            self.background_task = None
            logger.info("Background document processor stopped")
    
    async def _background_processor(self):
        """Background task to process documents from the queue"""
        while True:
            try:
                # Get document from queue
                document_info = await self.processing_queue.get()
                
                # Process the document
                await self._process_document_background(document_info)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background processing error: {e}")
    
    async def upload_and_process_document(
        self, 
        file_content: bytes, 
        filename: str,
        user_id: str = "anonymous",
        background: bool = False
    ) -> Dict[str, Any]:
        """
        Upload and process a document dynamically
        
        Args:
            file_content: Binary content of the file
            filename: Original filename
            user_id: ID of the user uploading
            background: Whether to process in background
            
        Returns:
            Processing result with document ID and status
        """
        try:
            # Generate unique document ID
            document_id = str(uuid.uuid4())
            
            # Save file with unique name
            file_extension = Path(filename).suffix.lower()
            unique_filename = f"{document_id}_{filename}"
            file_path = self.upload_dir / unique_filename
            
            # Write file content
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            # Generate document hash for deduplication
            content_hash = hashlib.md5(file_content).hexdigest()
            
            # Check for duplicates
            if await self._check_duplicate(content_hash):
                os.remove(file_path)
                return {
                    "document_id": document_id,
                    "status": "duplicate",
                    "message": "Document already exists in the system",
                    "filename": filename
                }
            
            # Register document
            document_info = {
                "document_id": document_id,
                "filename": filename,
                "file_path": str(file_path),
                "file_size": len(file_content),
                "content_hash": content_hash,
                "user_id": user_id,
                "upload_time": datetime.now(),
                "status": "uploaded",
                "processing_started": None,
                "processing_completed": None
            }
            
            self.document_registry[document_id] = document_info
            
            if background:
                # Add to processing queue for background processing
                await self.processing_queue.put(document_info)
                return {
                    "document_id": document_id,
                    "status": "queued",
                    "message": "Document queued for background processing",
                    "filename": filename,
                    "file_size": len(file_content)
                }
            else:
                # Process immediately
                result = await self._process_document_immediate(document_info)
                return result
                
        except Exception as e:
            logger.error(f"Upload and process error: {e}")
            raise
    
    async def _process_document_immediate(self, document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process document immediately and return results"""
        document_id = document_info["document_id"]
        file_path = document_info["file_path"]
        
        try:
            # Update status
            document_info["status"] = "processing"
            document_info["processing_started"] = datetime.now()
            
            logger.info(f"Processing document {document_id}: {document_info['filename']}")
            
            # Process through pipeline
            result = await self._run_processing_pipeline(file_path, document_info)
            
            # Update status
            document_info["status"] = "completed"
            document_info["processing_completed"] = datetime.now()
            document_info["processing_result"] = result
            
            # Move file to processed directory
            processed_path = self.processed_dir / Path(file_path).name
            shutil.move(file_path, processed_path)
            document_info["processed_path"] = str(processed_path)
            
            return {
                "document_id": document_id,
                "status": "completed",
                "filename": document_info["filename"],
                "chunks_created": result["chunks_created"],
                "processing_time": result["processing_time"],
                "embeddings_generated": result["embeddings_generated"],
                "file_size": document_info["file_size"]
            }
            
        except Exception as e:
            # Update status to failed
            document_info["status"] = "failed"
            document_info["error"] = str(e)
            
            logger.error(f"Document processing failed for {document_id}: {e}")
            raise
    
    async def _process_document_background(self, document_info: Dict[str, Any]):
        """Process document in background"""
        try:
            await self._process_document_immediate(document_info)
            logger.info(f"Background processing completed for {document_info['document_id']}")
        except Exception as e:
            logger.error(f"Background processing failed for {document_info['document_id']}: {e}")
    
    async def _run_processing_pipeline(self, file_path: str, document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete document processing pipeline"""
        start_time = time.time()
        
        # Step 1: Document ingestion and chunking
        logger.info(f"Step 1: Processing document {file_path}")
        document_result = await self.processor.process_document(file_path)
        
        chunks = document_result["chunks"]
        metadata = document_result["metadata"]
        
        # Step 2: Generate embeddings
        logger.info(f"Step 2: Generating embeddings for {len(chunks)} chunks")
        hierarchical_embeddings = await self.embedder.generate_hierarchical_embeddings(
            chunks, metadata
        )
        
        # Step 3: Store in vector database
        logger.info(f"Step 3: Storing embeddings in vector database")
        storage_result = await self.vector_store.store_document_embeddings(
            chunks, hierarchical_embeddings
        )
        
        processing_time = time.time() - start_time
        
        return {
            "chunks_created": len(chunks),
            "embeddings_generated": len(hierarchical_embeddings.chunk_embeddings),
            "section_embeddings": len(hierarchical_embeddings.section_embeddings),
            "storage_result": storage_result,
            "processing_time": round(processing_time, 2),
            "metadata": {
                "file_type": metadata.file_type,
                "total_words": metadata.total_words,
                "total_pages": metadata.total_pages
            }
        }
    
    async def _check_duplicate(self, content_hash: str) -> bool:
        """Check if document already exists based on content hash"""
        # Check in registry first
        for doc_info in self.document_registry.values():
            if doc_info.get("content_hash") == content_hash:
                return True
        
        # Could also check in database
        # For now, just check registry
        return False
    
    async def get_document_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get the processing status of a document"""
        document_info = self.document_registry.get(document_id)
        
        if not document_info:
            return None
        
        return {
            "document_id": document_id,
            "filename": document_info["filename"],
            "status": document_info["status"],
            "upload_time": document_info["upload_time"].isoformat(),
            "processing_started": document_info["processing_started"].isoformat() if document_info["processing_started"] else None,
            "processing_completed": document_info["processing_completed"].isoformat() if document_info["processing_completed"] else None,
            "file_size": document_info["file_size"],
            "processing_result": document_info.get("processing_result"),
            "error": document_info.get("error")
        }
    
    async def list_documents(self, user_id: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List documents with optional filtering"""
        documents = []
        
        for doc_info in self.document_registry.values():
            # Apply filters
            if user_id and doc_info["user_id"] != user_id:
                continue
            if status and doc_info["status"] != status:
                continue
            
            documents.append({
                "document_id": doc_info["document_id"],
                "filename": doc_info["filename"],
                "status": doc_info["status"],
                "upload_time": doc_info["upload_time"].isoformat(),
                "file_size": doc_info["file_size"],
                "user_id": doc_info["user_id"]
            })
        
        return sorted(documents, key=lambda x: x["upload_time"], reverse=True)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the system"""
        document_info = self.document_registry.get(document_id)
        
        if not document_info:
            return False
        
        try:
            # Remove files
            if os.path.exists(document_info["file_path"]):
                os.remove(document_info["file_path"])
            
            if "processed_path" in document_info and os.path.exists(document_info["processed_path"]):
                os.remove(document_info["processed_path"])
            
            # Remove from registry
            del self.document_registry[document_id]
            
            # TODO: Remove from vector database (need to implement)
            logger.info(f"Document {document_id} deleted")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        total_docs = len(self.document_registry)
        completed = sum(1 for d in self.document_registry.values() if d["status"] == "completed")
        processing = sum(1 for d in self.document_registry.values() if d["status"] == "processing")
        queued = sum(1 for d in self.document_registry.values() if d["status"] == "queued")
        failed = sum(1 for d in self.document_registry.values() if d["status"] == "failed")
        
        return {
            "total_documents": total_docs,
            "completed": completed,
            "processing": processing,
            "queued": queued,
            "failed": failed,
            "queue_size": self.processing_queue.qsize(),
            "background_processor_running": self.background_task is not None and not self.background_task.done()
        }

# Create singleton instance
dynamic_processor = DynamicDocumentProcessor()
