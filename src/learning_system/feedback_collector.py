"""
Feedback Collection and Learning System for Intelligent Document Q&A
Implements Phase 3: Learning and Adaptation capabilities
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import os
import uuid
from pathlib import Path
import asyncio
import numpy as np
from loguru import logger


class FeedbackType(Enum):
    """Types of feedback that can be collected"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"  # 1-5 stars
    DETAILED = "detailed"  # Text feedback
    CORRECTION = "correction"  # User provides correct answer
    RELEVANCE = "relevance"  # Feedback on answer relevance
    COMPLETENESS = "completeness"  # Feedback on answer completeness


@dataclass
class UserFeedback:
    """Represents user feedback on a Q&A response"""
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    turn_id: str = ""
    question: str = ""
    answer: str = ""
    feedback_type: FeedbackType = FeedbackType.RATING
    feedback_value: Any = None  # Rating number, boolean, or text
    feedback_text: Optional[str] = None
    suggested_improvement: Optional[str] = None
    user_correction: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningPattern:
    """Represents a learned pattern from user feedback"""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: str = ""  # e.g., "question_type", "answer_style", "topic_preference"
    pattern_description: str = ""
    confidence_score: float = 0.0
    supporting_feedback_count: int = 0
    improvement_suggestion: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeedbackCollector:
    """Collects and manages user feedback"""
    
    def __init__(self, feedback_directory: str = "data/feedback"):
        self.feedback_directory = Path(feedback_directory)
        self.feedback_directory.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_directory / "user_feedback.json"
        self.feedback_history: List[UserFeedback] = []
        self._load_feedback()
    
    def _load_feedback(self):
        """Load existing feedback from disk"""
        try:
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.feedback_history = []
                for item in data.get('feedback', []):
                    feedback = UserFeedback(
                        feedback_id=item.get('feedback_id', str(uuid.uuid4())),
                        session_id=item.get('session_id', ''),
                        turn_id=item.get('turn_id', ''),
                        question=item.get('question', ''),
                        answer=item.get('answer', ''),
                        feedback_type=FeedbackType(item.get('feedback_type', 'rating')),
                        feedback_value=item.get('feedback_value'),
                        feedback_text=item.get('feedback_text'),
                        suggested_improvement=item.get('suggested_improvement'),
                        user_correction=item.get('user_correction'),
                        timestamp=datetime.fromisoformat(item.get('timestamp', datetime.now().isoformat())),
                        user_id=item.get('user_id'),
                        metadata=item.get('metadata', {})
                    )
                    self.feedback_history.append(feedback)
                
                logger.info(f"Loaded {len(self.feedback_history)} feedback entries")
                
        except Exception as e:
            logger.error(f"Error loading feedback: {e}")
            self.feedback_history = []
    
    def _save_feedback(self):
        """Save feedback to disk"""
        try:
            data = {
                'feedback': [
                    {
                        'feedback_id': fb.feedback_id,
                        'session_id': fb.session_id,
                        'turn_id': fb.turn_id,
                        'question': fb.question,
                        'answer': fb.answer,
                        'feedback_type': fb.feedback_type.value,
                        'feedback_value': fb.feedback_value,
                        'feedback_text': fb.feedback_text,
                        'suggested_improvement': fb.suggested_improvement,
                        'user_correction': fb.user_correction,
                        'timestamp': fb.timestamp.isoformat(),
                        'user_id': fb.user_id,
                        'metadata': fb.metadata
                    }
                    for fb in self.feedback_history
                ],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
    
    def collect_feedback(self, session_id: str, turn_id: str, question: str, 
                        answer: str, feedback_type: FeedbackType, 
                        feedback_value: Any, **kwargs) -> str:
        """Collect user feedback on a Q&A response"""
        try:
            feedback = UserFeedback(
                session_id=session_id,
                turn_id=turn_id,
                question=question,
                answer=answer,
                feedback_type=feedback_type,
                feedback_value=feedback_value,
                feedback_text=kwargs.get('feedback_text'),
                suggested_improvement=kwargs.get('suggested_improvement'),
                user_correction=kwargs.get('user_correction'),
                user_id=kwargs.get('user_id'),
                metadata=kwargs.get('metadata', {})
            )
            
            self.feedback_history.append(feedback)
            self._save_feedback()
            
            logger.info(f"Collected {feedback_type.value} feedback for turn {turn_id}")
            return feedback.feedback_id
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            return ""
    
    def get_feedback_for_session(self, session_id: str) -> List[UserFeedback]:
        """Get all feedback for a specific session"""
        return [fb for fb in self.feedback_history if fb.session_id == session_id]
    
    def get_feedback_for_turn(self, turn_id: str) -> List[UserFeedback]:
        """Get all feedback for a specific turn"""
        return [fb for fb in self.feedback_history if fb.turn_id == turn_id]
    
    def get_recent_feedback(self, days: int = 7) -> List[UserFeedback]:
        """Get feedback from the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [fb for fb in self.feedback_history if fb.timestamp >= cutoff_date]
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected feedback"""
        if not self.feedback_history:
            return {"total_feedback": 0}
        
        # Count by type
        type_counts = {}
        for fb in self.feedback_history:
            type_name = fb.feedback_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Calculate average ratings
        ratings = [fb.feedback_value for fb in self.feedback_history 
                  if fb.feedback_type == FeedbackType.RATING and isinstance(fb.feedback_value, (int, float))]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        # Count positive/negative feedback
        positive_feedback = len([fb for fb in self.feedback_history 
                               if fb.feedback_type == FeedbackType.THUMBS_UP or 
                               (fb.feedback_type == FeedbackType.RATING and 
                                isinstance(fb.feedback_value, (int, float)) and fb.feedback_value >= 4)])
        
        negative_feedback = len([fb for fb in self.feedback_history 
                               if fb.feedback_type == FeedbackType.THUMBS_DOWN or 
                               (fb.feedback_type == FeedbackType.RATING and 
                                isinstance(fb.feedback_value, (int, float)) and fb.feedback_value <= 2)])
        
        return {
            "total_feedback": len(self.feedback_history),
            "feedback_by_type": type_counts,
            "average_rating": round(avg_rating, 2),
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "satisfaction_rate": round(positive_feedback / len(self.feedback_history) * 100, 1) if self.feedback_history else 0,
            "improvement_suggestions": len([fb for fb in self.feedback_history if fb.suggested_improvement]),
            "user_corrections": len([fb for fb in self.feedback_history if fb.user_correction])
        }


# Global feedback collector instance
feedback_collector = FeedbackCollector()
