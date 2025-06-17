"""
Document processor package initialization
Exports main components for Phase 1: Document Processing Pipeline
"""

from .document_ingestion import document_processor, DocumentChunk, DocumentMetadata
from .embedding_generator import embedding_generator, HierarchicalEmbedding, EmbeddingResult
from .vector_database import vector_db
from .main_pipeline import document_pipeline

__all__ = [
    'document_processor',
    'embedding_generator', 
    'vector_db',
    'document_pipeline',
    'DocumentChunk',
    'DocumentMetadata',
    'HierarchicalEmbedding',
    'EmbeddingResult'
]
