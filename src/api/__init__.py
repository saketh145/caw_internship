"""
API package for Production Document Q&A System
Phase 4: Production Deployment & Advanced Features
"""

from .main import app
from .auth import AuthManager, User, UserRole
from .monitoring import SystemMonitor
from .models import *

__all__ = [
    'app',
    'AuthManager',
    'User', 
    'UserRole',
    'SystemMonitor',
]
