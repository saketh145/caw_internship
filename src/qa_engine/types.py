"""
Data types for the Q&A Engine
Contains shared data structures to avoid circular imports
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class QAResponse:
    """Response from the Q&A engine"""
    answer: str
    confidence_score: float
    sources: List[str]
    context_used: List[Dict[str, Any]]
    processing_time: float
    session_id: str
    turn_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
