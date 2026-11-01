try:
    from beanie import Document
except ImportError:
    class Document:  # type: ignore
        pass
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


class User(Document):
    email: EmailStr
    username: str
    hashed_password: str
    full_name: Optional[str] = None
    family_contact_email: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None

    class Settings:
        name = "users"
        indexes = ["email", "username"]


class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None
    family_contact_email: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    full_name: Optional[str] = None
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
