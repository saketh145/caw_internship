"""
Production API Server for Intelligent Document Q&A System
Phase 4: Production Deployment & Advanced Features
"""

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime, timedelta
import uuid
import json
from pathlib import Path

from ..qa_engine import qa_engine
from ..document_processor.main_pipeline import document_pipeline
from ..document_processor.dynamic_processor import dynamic_processor
from ..learning_system import feedback_collector, learning_engine
from .auth import AuthManager, User, UserRole
from .models import *
from .monitoring import SystemMonitor
from loguru import logger

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Document Q&A API",
    description="Production-ready API for document ingestion, Q&A, and learning",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
auth_manager = AuthManager()
security = HTTPBearer()
system_monitor = SystemMonitor()

# Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user"""
    try:
        user = await auth_manager.verify_token(credentials.credentials)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Health check endpoints
@app.get("/health", tags=["System"])
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0.0",
        "components": {
            "database": "operational",
            "qa_engine": "operational",
            "learning_system": "operational"
        }
    }

@app.get("/metrics", tags=["System"])
async def get_metrics():
    """Get system metrics"""
    return await system_monitor.get_metrics()

# Authentication endpoints
@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(credentials: LoginRequest):
    """User login"""
    try:
        token_data = auth_manager.authenticate_user(
            credentials.username, 
            credentials.password
        )
        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        return token_data
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )

@app.post("/auth/register", response_model=UserResponse, tags=["Authentication"])
async def register(user_data: CreateUserRequest):
    """Register new user"""
    try:
        user = await auth_manager.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            role=user_data.role or UserRole.USER
        )
        return UserResponse.from_user(user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User registration failed"
        )

# Document management endpoints
@app.post("/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    background_processing: bool = False,
    category: Optional[str] = None,
    tags: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Upload and process a document with dynamic chunking and embedding storage
    
    Args:
        file: Document file to upload
        background_processing: Whether to process in background
        category: Document category (optional)
        tags: Document tags (optional)
        current_user: Authenticated user
    """
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.txt', '.md', '.html'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file_extension}. Supported: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )
        
        # Process document using dynamic processor
        result = await dynamic_processor.upload_and_process_document(
            file_content=file_content,
            filename=file.filename,
            user_id=current_user.id,
            background=background_processing
        )
        
        # Track upload
        await system_monitor.track_document_upload(
            user_id=current_user.id,
            filename=file.filename,
            file_size=len(file_content),
            chunks_created=result.get('chunks_created', 0)
        )
        
        return DocumentUploadResponse(
            document_id=result['document_id'],
            filename=file.filename,
            status=result['status'],
            chunks_created=result.get('chunks_created', 0),
            processing_time=result.get('processing_time', 0),
            upload_time=datetime.now(),
            message=result.get('message', 'Document processed successfully')
        )
        
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )

@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    status_filter: Optional[str] = None,
    user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """List documents with dynamic filtering"""
    try:
        # Get documents from dynamic processor
        documents = await dynamic_processor.list_documents(
            user_id=user_id,
            status=status_filter
        )
        
        # Apply pagination
        paginated_docs = documents[skip:skip + limit]
        
        # Convert to response format
        document_list = []
        for doc in paginated_docs:
            document_list.append(DocumentInfo(
                document_id=doc["document_id"],
                filename=doc["filename"],
                upload_date=doc["upload_time"],
                file_size=doc["file_size"],
                status=doc["status"],
                user_id=doc["user_id"]
            ))
        
        return document_list
        
    except Exception as e:
        logger.error(f"Document listing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list documents"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents"
        )

@app.get("/documents/{document_id}/status", response_model=Dict[str, Any], tags=["Documents"])
async def get_document_status(
    document_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get the processing status of a specific document"""
    try:
        status = await dynamic_processor.get_document_status(document_id)
        
        if not status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get document status"
        )

@app.delete("/documents/{document_id}", tags=["Documents"])
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a document from the system"""
    try:
        success = await dynamic_processor.delete_document(document_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return {"message": "Document deleted successfully", "document_id": document_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )

@app.get("/documents/processing/stats", response_model=Dict[str, Any], tags=["Documents"])
async def get_processing_stats(
    current_user: User = Depends(get_current_user)
):
    """Get document processing statistics"""
    try:
        stats = dynamic_processor.get_processing_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting processing stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get processing statistics"
        )

@app.post("/documents/processing/start-background", tags=["Documents"])
async def start_background_processing(
    current_user: User = Depends(get_current_user)
):
    """Start the background document processing task"""
    try:
        if current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        await dynamic_processor.start_background_processor()
        return {"message": "Background processing started"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting background processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start background processing"
        )

@app.post("/documents/processing/stop-background", tags=["Documents"])
async def stop_background_processing(
    current_user: User = Depends(get_current_user)
):
    """Stop the background document processing task"""
    try:
        if current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        await dynamic_processor.stop_background_processor()
        return {"message": "Background processing stopped"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping background processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop background processing"
        )

# Q&A endpoints
@app.post("/qa/ask", response_model=QAResponse, tags=["Q&A"])
async def ask_question(
    request: QARequest,
    current_user: User = Depends(get_current_user)
):
    """Ask a question and get an answer"""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"api_session_{current_user.id}_{uuid.uuid4().hex[:8]}"
          # Get answer from Q&A engine
        response = await qa_engine.answer_question(
            question=request.question,
            session_id=session_id
        )
          # Track question
        await system_monitor.track_question(
            user_id=current_user.id,
            question=request.question,
            response_time=response.processing_time,
            confidence=response.confidence_score
        )
        
        return QAResponse(
            answer=response.answer,
            confidence_score=response.confidence_score,
            sources=[
                SourceInfo(
                    document_id=f"source_{i}",
                    content=src if isinstance(src, str) else str(src),
                    relevance_score=0.8,  # Default relevance
                    metadata={}
                ) for i, src in enumerate(response.sources)
            ],
            session_id=session_id,
            processing_time=response.processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Q&A error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question processing failed: {str(e)}"
        )

@app.post("/qa/feedback", response_model=FeedbackResponse, tags=["Q&A"])
async def submit_feedback(
    feedback: FeedbackRequest,
    current_user: User = Depends(get_current_user)
):
    """Submit feedback for a Q&A response"""
    try:
        # Collect feedback through the learning system
        feedback_id = await qa_engine.collect_user_feedback(
            turn_id=feedback.turn_id,
            feedback_type=feedback.feedback_type,
            value=feedback.value,
            comment=feedback.comment
        )
        
        # Track feedback
        await system_monitor.track_feedback(
            user_id=current_user.id,
            feedback_type=feedback.feedback_type,
            value=feedback.value
        )
        
        return FeedbackResponse(
            feedback_id=feedback_id,
            status="collected",
            message="Feedback collected successfully"
        )
        
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback submission failed: {str(e)}"
        )

# Learning system endpoints
@app.get("/learning/patterns", response_model=List[Dict[str, Any]], tags=["Learning"])
async def get_learning_patterns(
    current_user: User = Depends(get_current_user)
):
    """Get discovered learning patterns"""
    try:
        patterns = learning_engine.get_patterns()
        return [
            {
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type,
                "description": pattern.pattern_description,
                "confidence": pattern.confidence_score,
                "occurrences": pattern.supporting_feedback_count,
                "discovered_at": pattern.created_at.isoformat()
            }
            for pattern in patterns
        ]
    except Exception as e:
        logger.error(f"Learning patterns error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve learning patterns"
        )

@app.post("/learning/analyze", tags=["Learning"])
async def trigger_learning_analysis(
    current_user: User = Depends(get_current_user)
):
    """Trigger learning analysis"""
    try:
        # Check if user has admin privileges
        if current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
        
        patterns, rules = await learning_engine.analyze_feedback_patterns()
        
        return {
            "status": "completed",
            "patterns_discovered": len(patterns),
            "rules_generated": len(rules),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Learning analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Learning analysis failed"
        )

# Analytics endpoints
@app.get("/analytics/dashboard", response_model=Dict[str, Any], tags=["Analytics"])
async def get_dashboard_data(
    current_user: User = Depends(get_current_user)
):
    """Get dashboard analytics data"""
    try:
        return await system_monitor.get_dashboard_data()
    except Exception as e:
        logger.error(f"Dashboard data error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard data"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Intelligent Document Q&A API...")
    
    # Initialize background processor
    await dynamic_processor.start_background_processor()
    logger.info("Background document processor initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Intelligent Document Q&A API...")
    
    # Stop background processor
    await dynamic_processor.stop_background_processor()
    logger.info("Background document processor stopped")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
