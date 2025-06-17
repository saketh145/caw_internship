"""
Authentication and authorization system
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path
import uuid

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from .models import UserRole

# Configuration
SECRET_KEY = "your-secret-key-change-in-production"  # Should be in environment variables
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(BaseModel):
    id: str
    username: str
    email: str
    role: UserRole
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True

class UserInDB(User):
    hashed_password: str

class AuthManager:
    """Manages user authentication and authorization"""
    
    def __init__(self):
        self.users_file = Path("data/users.json")
        self.users_file.parent.mkdir(exist_ok=True)
        self.users: Dict[str, UserInDB] = {}
        self._load_users()
        self._create_default_admin()
    
    def _load_users(self):
        """Load users from file"""
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r') as f:
                    data = json.load(f)
                    for user_data in data.values():
                        user_data['created_at'] = datetime.fromisoformat(user_data['created_at'])
                        if user_data.get('last_login'):
                            user_data['last_login'] = datetime.fromisoformat(user_data['last_login'])
                        self.users[user_data['username']] = UserInDB(**user_data)
            except Exception as e:
                print(f"Error loading users: {e}")
    
    def _save_users(self):
        """Save users to file"""
        try:
            data = {}
            for username, user in self.users.items():
                user_dict = user.dict()
                user_dict['created_at'] = user_dict['created_at'].isoformat()
                if user_dict.get('last_login'):
                    user_dict['last_login'] = user_dict['last_login'].isoformat()
                data[username] = user_dict
            
            with open(self.users_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving users: {e}")
    
    def _create_default_admin(self):
        """Create default admin user if none exists"""
        if not any(user.role == UserRole.ADMIN for user in self.users.values()):
            admin_user = UserInDB(
                id=str(uuid.uuid4()),
                username="admin",
                email="admin@example.com",
                role=UserRole.ADMIN,
                created_at=datetime.now(),
                hashed_password=self.get_password_hash("admin123"),
                is_active=True
            )
            self.users["admin"] = admin_user
            self._save_users()
            print("Created default admin user: admin/admin123")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    def get_user(self, username: str) -> Optional[UserInDB]:
        """Get user by username"""
        return self.users.get(username)
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return token data"""
        user = self.get_user(username)
        if not user or not user.is_active:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        user.last_login = datetime.now()
        self._save_users()
        
        # Generate token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = self.create_access_token(
            data={"sub": user.username, "user_id": user.id, "role": user.role},
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user_id": user.id,
            "username": user.username,
            "role": user.role
        }
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    async def verify_token(self, token: str) -> Optional[User]:
        """Verify JWT token and return user"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                return None
            
            user = self.get_user(username)
            if user is None or not user.is_active:
                return None
            
            return User(
                id=user.id,
                username=user.username,
                email=user.email,
                role=user.role,
                created_at=user.created_at,
                last_login=user.last_login,
                is_active=user.is_active
            )
        except JWTError:
            return None
    
    async def create_user(
        self, 
        username: str, 
        email: str, 
        password: str, 
        role: UserRole = UserRole.USER
    ) -> User:
        """Create a new user"""
        if username in self.users:
            raise ValueError("Username already exists")
        
        # Check if email already exists
        if any(user.email == email for user in self.users.values()):
            raise ValueError("Email already exists")
        
        user = UserInDB(
            id=str(uuid.uuid4()),
            username=username,
            email=email,
            role=role,
            created_at=datetime.now(),
            hashed_password=self.get_password_hash(password),
            is_active=True
        )
        
        self.users[username] = user
        self._save_users()
        
        return User(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role,
            created_at=user.created_at,
            last_login=user.last_login,
            is_active=user.is_active
        )
    
    def update_user_role(self, username: str, new_role: UserRole) -> bool:
        """Update user role"""
        user = self.get_user(username)
        if not user:
            return False
        
        user.role = new_role
        self._save_users()
        return True
    
    def deactivate_user(self, username: str) -> bool:
        """Deactivate a user"""
        user = self.get_user(username)
        if not user:
            return False
        
        user.is_active = False
        self._save_users()
        return True
    
    def list_users(self) -> list[User]:
        """List all users"""
        return [
            User(
                id=user.id,
                username=user.username,
                email=user.email,
                role=user.role,
                created_at=user.created_at,
                last_login=user.last_login,
                is_active=user.is_active
            )
            for user in self.users.values()
        ]
