"""
Pydantic models for the API
"""

from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

# User and Authentication Models
class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    role: Optional[UserRole] = UserRole.USER

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    username: str
    role: UserRole

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: UserRole
    created_at: datetime
    last_login: Optional[datetime] = None
    
    @classmethod
    def from_user(cls, user):
        return cls(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role,
            created_at=user.created_at,
            last_login=user.last_login
        )

# Document Models
class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks_created: int
    processing_time: float
    upload_time: datetime
    message: Optional[str] = None

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    upload_date: str  # ISO format string
    file_size: int
    status: str
    user_id: str
    category: Optional[str] = None
    tags: List[str] = []

# Q&A Models
class QARequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None
    context: Optional[str] = None
    max_sources: Optional[int] = Field(default=5, ge=1, le=20)

class SourceInfo(BaseModel):
    document_id: str
    content: str
    relevance_score: float
    metadata: Dict[str, Any] = {}

class QAResponse(BaseModel):
    answer: str
    confidence_score: float
    sources: List[SourceInfo]
    session_id: str
    processing_time: float
    timestamp: datetime

# Feedback Models
class FeedbackType(str, Enum):
    RATING = "rating"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    DETAILED = "detailed"

class FeedbackRequest(BaseModel):
    turn_id: str
    feedback_type: FeedbackType
    value: Union[int, bool, str]
    comment: Optional[str] = None

class FeedbackResponse(BaseModel):
    feedback_id: str
    status: str
    message: str

# Analytics Models
class UsageStats(BaseModel):
    total_questions: int
    total_documents: int
    total_users: int
    avg_response_time: float
    avg_confidence_score: float
    questions_per_day: List[Dict[str, Any]]
    top_questions: List[str]
    document_usage: List[Dict[str, Any]]

class LearningStats(BaseModel):
    total_feedback: int
    patterns_discovered: int
    rules_generated: int
    learning_effectiveness: float
    feedback_distribution: Dict[str, int]

class SystemStats(BaseModel):
    uptime: str
    memory_usage: float
    cpu_usage: float
    database_size: int
    response_times: List[float]
    error_rate: float

# Bulk Operations
class BulkUploadRequest(BaseModel):
    files: List[str]  # File paths or URLs
    category: Optional[str] = None
    tags: List[str] = []

class BulkUploadResponse(BaseModel):
    upload_id: str
    total_files: int
    processed_files: int
    failed_files: int
    status: str
    errors: List[str] = []

# Search and Filtering
class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = {}
    limit: Optional[int] = Field(default=10, ge=1, le=100)
    offset: Optional[int] = Field(default=0, ge=0)

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int
    query: str
    processing_time: float

# Configuration Models
class SystemConfig(BaseModel):
    max_upload_size: int
    supported_formats: List[str]
    rate_limits: Dict[str, int]
    feature_flags: Dict[str, bool]

# Health and Monitoring
class HealthStatus(BaseModel):
    status: str
    timestamp: datetime
    version: str
    components: Dict[str, str]
    
class MetricsResponse(BaseModel):
    system_stats: SystemStats
    usage_stats: UsageStats
    learning_stats: LearningStats
    timestamp: datetime
