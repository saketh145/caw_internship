"""
Learning System for Intelligent Document Q&A
Phase 3: Learning and Adaptation capabilities
"""

from .feedback_collector import (
    UserFeedback,
    FeedbackType,
    FeedbackCollector,
    feedback_collector
)

from .learning_engine import (
    LearningPattern,
    AdaptationRule,
    LearningEngine,
    learning_engine
)

from .response_adapter import (
    AdaptedResponse,
    ResponseAdapter,
    response_adapter
)

__all__ = [
    # Feedback Collection
    'UserFeedback',
    'FeedbackType',
    'FeedbackCollector',
    'feedback_collector',
    
    # Learning Engine
    'LearningPattern',
    'AdaptationRule',
    'LearningEngine',
    'learning_engine',
    
    # Response Adaptation
    'AdaptedResponse',
    'ResponseAdapter',
    'response_adapter'
]
