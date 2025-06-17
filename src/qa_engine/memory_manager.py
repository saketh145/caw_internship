"""
Memory Management System for the Q&A Engine
Handles conversation history, context tracking, and memory persistence
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from uuid import uuid4
import asyncio

from ..config import settings
from loguru import logger


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation"""
    turn_id: str
    session_id: str
    timestamp: datetime
    user_question: str
    system_answer: str
    context_used: List[Dict[str, Any]]  # Retrieved documents used
    confidence_score: float
    sources: List[str]  # Document sources
    metadata: Dict[str, Any]


@dataclass
class ConversationSession:
    """Represents a complete conversation session"""
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    last_active: datetime
    turns: List[ConversationTurn]
    session_metadata: Dict[str, Any]


class ConversationMemory:
    """Manages conversation history and memory"""
    
    def __init__(self, memory_dir: str = "data/memory"):
        self.memory_dir = memory_dir
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.max_session_age_hours = settings.max_session_age_hours if hasattr(settings, 'max_session_age_hours') else 24
        self.max_turns_per_session = settings.max_turns_per_session if hasattr(settings, 'max_turns_per_session') else 50
        
        # Create memory directory
        os.makedirs(memory_dir, exist_ok=True)
        
        # Load existing sessions
        self._load_sessions()
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create a new conversation session"""
        session_id = str(uuid4())
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_active=datetime.now(),
            turns=[],
            session_metadata={}
        )
        
        self.active_sessions[session_id] = session
        self._persist_session(session)
        
        logger.info(f"Created new conversation session: {session_id}")
        return session_id
    
    def add_turn(self, session_id: str, user_question: str, system_answer: str,
                 context_used: List[Dict[str, Any]], confidence_score: float,
                 sources: List[str], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a new turn to a conversation session"""
        
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found, creating new session")
            session_id = self.create_session()
        
        session = self.active_sessions[session_id]
        turn_id = str(uuid4())
        
        turn = ConversationTurn(
            turn_id=turn_id,
            session_id=session_id,
            timestamp=datetime.now(),
            user_question=user_question,
            system_answer=system_answer,
            context_used=context_used,
            confidence_score=confidence_score,
            sources=sources,
            metadata=metadata or {}
        )
        
        session.turns.append(turn)
        session.last_active = datetime.now()
        
        # Limit session length
        if len(session.turns) > self.max_turns_per_session:
            session.turns = session.turns[-self.max_turns_per_session:]
        
        self._persist_session(session)
        
        logger.info(f"Added turn {turn_id} to session {session_id}")
        return turn_id
    
    def get_session_context(self, session_id: str, max_turns: int = 5) -> List[ConversationTurn]:
        """Get recent conversation context for a session"""
        if session_id not in self.active_sessions:
            return []
        
        session = self.active_sessions[session_id]
        return session.turns[-max_turns:] if session.turns else []
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of the conversation session"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        if not session.turns:
            return {
                "session_id": session_id,
                "turn_count": 0,
                "created_at": session.created_at.isoformat(),
                "last_active": session.last_active.isoformat(),
                "topics": [],
                "avg_confidence": 0.0
            }
        
        # Calculate average confidence
        avg_confidence = sum(turn.confidence_score for turn in session.turns) / len(session.turns)
          # Extract topics (simplified - could be more sophisticated)
        topics = list(set(" ".join(turn.user_question.split()[:3]) for turn in session.turns[-5:]))
        
        return {
            "session_id": session_id,
            "turn_count": len(session.turns),
            "created_at": session.created_at.isoformat(),
            "last_active": session.last_active.isoformat(),
            "topics": topics[:5],  # Top 5 recent topics
            "avg_confidence": round(avg_confidence, 2)
        }
    
    def search_conversation_history(self, query: str, max_results: int = 10) -> List[ConversationTurn]:
        """Search through conversation history"""
        all_turns = []
        for session in self.active_sessions.values():
            all_turns.extend(session.turns)
        
        # Simple keyword search (could be enhanced with semantic search)
        query_words = query.lower().split()
        matching_turns = []
        
        for turn in all_turns:
            question_text = turn.user_question.lower()
            answer_text = turn.system_answer.lower()
            
            # Count matching words
            question_matches = sum(1 for word in query_words if word in question_text)
            answer_matches = sum(1 for word in query_words if word in answer_text)
            total_matches = question_matches + answer_matches
            
            if total_matches > 0:
                matching_turns.append((turn, total_matches))
        
        # Sort by relevance
        matching_turns.sort(key=lambda x: x[1], reverse=True)
        return [turn for turn, _ in matching_turns[:max_results]]
    
    def cleanup_old_sessions(self):
        """Remove old inactive sessions"""
        cutoff_time = datetime.now() - timedelta(hours=self.max_session_age_hours)
        sessions_to_remove = []
        
        for session_id, session in self.active_sessions.items():
            if session.last_active < cutoff_time:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            self._archive_session(session_id)
            del self.active_sessions[session_id]
        
        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
    
    def _persist_session(self, session: ConversationSession):
        """Persist session to disk"""
        try:
            session_file = os.path.join(self.memory_dir, f"session_{session.session_id}.json")
            session_data = {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "created_at": session.created_at.isoformat(),
                "last_active": session.last_active.isoformat(),
                "session_metadata": session.session_metadata,
                "turns": [
                    {
                        "turn_id": turn.turn_id,
                        "session_id": turn.session_id,
                        "timestamp": turn.timestamp.isoformat(),
                        "user_question": turn.user_question,
                        "system_answer": turn.system_answer,
                        "context_used": turn.context_used,
                        "confidence_score": turn.confidence_score,
                        "sources": turn.sources,
                        "metadata": turn.metadata
                    }
                    for turn in session.turns
                ]
            }
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error persisting session {session.session_id}: {e}")
    
    def _load_sessions(self):
        """Load existing sessions from disk"""
        try:
            if not os.path.exists(self.memory_dir):
                return
            
            for filename in os.listdir(self.memory_dir):
                if filename.startswith("session_") and filename.endswith(".json"):
                    session_file = os.path.join(self.memory_dir, filename)
                    
                    try:
                        with open(session_file, 'r', encoding='utf-8') as f:
                            session_data = json.load(f)
                        
                        # Convert back to objects
                        turns = []
                        for turn_data in session_data["turns"]:
                            turn = ConversationTurn(
                                turn_id=turn_data["turn_id"],
                                session_id=turn_data["session_id"],
                                timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                                user_question=turn_data["user_question"],
                                system_answer=turn_data["system_answer"],
                                context_used=turn_data["context_used"],
                                confidence_score=turn_data["confidence_score"],
                                sources=turn_data["sources"],
                                metadata=turn_data["metadata"]
                            )
                            turns.append(turn)
                        
                        session = ConversationSession(
                            session_id=session_data["session_id"],
                            user_id=session_data["user_id"],
                            created_at=datetime.fromisoformat(session_data["created_at"]),
                            last_active=datetime.fromisoformat(session_data["last_active"]),
                            turns=turns,
                            session_metadata=session_data["session_metadata"]
                        )
                        
                        self.active_sessions[session.session_id] = session
                        
                    except Exception as e:
                        logger.error(f"Error loading session from {filename}: {e}")
            
            logger.info(f"Loaded {len(self.active_sessions)} conversation sessions")
            
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
    
    def _archive_session(self, session_id: str):
        """Archive an old session"""
        try:
            session_file = os.path.join(self.memory_dir, f"session_{session_id}.json")
            archive_file = os.path.join(self.memory_dir, "archived", f"session_{session_id}.json")
            
            os.makedirs(os.path.dirname(archive_file), exist_ok=True)
            
            if os.path.exists(session_file):
                os.rename(session_file, archive_file)
                
        except Exception as e:
            logger.error(f"Error archiving session {session_id}: {e}")


# Global memory instance
conversation_memory = ConversationMemory()
