"""
Q&A Engine Package
Provides context-aware question answering with memory and evaluation capabilities
"""

from .memory_manager import conversation_memory, ConversationTurn, ConversationSession
from .types import QAResponse
from .qa_engine import qa_engine, ContextAwareQAEngine
from .answer_evaluator import answer_evaluator, EvaluationResult, EvaluationMetrics

__all__ = [
    'conversation_memory',
    'ConversationTurn', 
    'ConversationSession',
    'qa_engine',
    'QAResponse',
    'ContextAwareQAEngine',
    'answer_evaluator',
    'EvaluationResult',
    'EvaluationMetrics'
]
