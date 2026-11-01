import os
from datetime import datetime, timedelta
from typing import Optional
import bcrypt
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "echoshield-super-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))  # 24 hours

security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8")
    )


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    from app.core.database import User
    token = credentials.credentials

    # Support demo mode access seamlessly
    if token.startswith("demo_user_token_") or token == "demo_token":
        try:
            demo_user = await User.find_one({"email": "demo@echoshield.ai"})
            if demo_user:
                return demo_user
        except Exception:
            pass
        from app.core.mock_db import MockUser
        demo = MockUser(
            user_id="demo_user_1",
            email="demo@echoshield.ai",
            username="DemoUser",
            full_name="Security Operator",
        )
        try:
            await demo.insert()
        except Exception:
            pass
        return demo

    payload = decode_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    if User is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    try:
        user = await User.get(user_id)
    except Exception:
        user = None
        
    if not user:
        from app.core.mock_db import MockUser
        user = await MockUser.get(user_id)
        if not user:
            # Recreate session user from verified JWT payload so stateless serverless instances never 404
            user = MockUser(
                user_id=str(user_id),
                email=payload.get("email", f"user_{str(user_id)[:8]}@echoshield.ai"),
                username=payload.get("username", "EchoUser"),
                full_name=payload.get("name", "EchoShield User"),
            )
            try:
                await user.insert()
            except Exception:
                pass
    return user


async def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_optional)):
    """Extract authenticated user if bearer token is present; otherwise returns None."""
    if not credentials or not credentials.credentials:
        return None
    try:
        return await get_current_user(credentials)
    except Exception:
        return None
