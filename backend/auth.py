from datetime import datetime
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from database import get_db
from models import User, UserRole
from config import get_settings

# Security configuration
settings = get_settings()
EXTERNAL_JWT_SECRET = settings.external_jwt_secret  # Shared secret from main system
ALGORITHM = "HS256"

# HTTP Bearer token scheme
security = HTTPBearer()


def decode_external_jwt(token: str) -> dict:
    """
    Decode and verify JWT token from the main system

    The token should contain:
    - user_id: The external user's ID from main system
    - username: The user's username
    - exp: Expiration timestamp (optional)
    """
    try:
        payload = jwt.decode(token, EXTERNAL_JWT_SECRET, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        print(f"❌ JWT Validation Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def sync_user_from_jwt(db: Session, external_user_id: str, username: str) -> User:
    """
    Sync user from JWT payload to local database

    If user doesn't exist, create it.
    If user exists, update username if changed.

    Args:
        db: Database session
        external_user_id: User ID from main system
        username: Username from main system

    Returns:
        User object from local database
    """
    # Check if user already exists
    user = db.query(User).filter(User.external_user_id == external_user_id).first()

    if user:
        # Update username if changed
        if user.username != username:
            user.username = username
            user.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(user)
            print(f"✅ Updated user: {username} (external_id: {external_user_id})")
    else:
        # Create new user
        user = User(
            external_user_id=external_user_id,
            username=username,
            role=UserRole.USER,  # Default role
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"✅ Created new user: {username} (external_id: {external_user_id})")

    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Get the current authenticated user from external JWT token

    This function:
    1. Validates the JWT token from the main system
    2. Extracts user_id and username
    3. Auto-creates/updates user in local database
    4. Returns the local user object
    """
    token = credentials.credentials
    payload = decode_external_jwt(token)

    # Extract user info from JWT payload
    external_user_id = payload.get("user_id")
    username = payload.get("username")

    if not external_user_id or not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user_id or username",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Convert user_id to string for consistency
    external_user_id_str = str(external_user_id)

    # Sync user from JWT to local database
    user = sync_user_from_jwt(db, external_user_id_str, username)

    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get the current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    return current_user


def require_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    """Require the current user to be an admin"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user