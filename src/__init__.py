"""
Intelligent Document Q&A System with Memory and Learning
Main package initialization
"""

from .config import settings
from .document_processor import document_pipeline
from .qa_engine import qa_engine, conversation_memory, answer_evaluator
from .learning_system import feedback_collector, learning_engine, response_adapter

__version__ = "1.0.0"
__all__ = [
    'settings', 
    'document_pipeline', 
    'qa_engine', 
    'conversation_memory', 
    'answer_evaluator',
    'feedback_collector',
    'learning_engine',
    'response_adapter'
]
