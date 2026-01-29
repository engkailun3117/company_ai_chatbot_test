"""
Simplified Authentication for Internal Testing Environment

This module provides a simplified authentication mechanism for internal testing.
Instead of JWT tokens, users simply provide their user_id via a header.
If the user doesn't exist, they are automatically created.
"""

from datetime import datetime
from typing import Optional
from urllib.parse import unquote
from fastapi import Depends, HTTPException, status, Header
from sqlalchemy.orm import Session
from database import get_db
from models_test import UserTest, UserRoleTest


def get_or_create_test_user(
    x_user_id: str = Header(..., description="User ID for testing"),
    x_display_name: Optional[str] = Header(None, description="Optional display name"),
    db: Session = Depends(get_db)
) -> UserTest:
    """
    Get or create a test user based on the provided user_id header.

    This simplified authentication:
    1. Receives user_id from X-User-ID header
    2. Looks up user in users_test table
    3. Creates new user if not found
    4. Updates display_name if provided and different

    Args:
        x_user_id: User ID from header (required)
        x_display_name: Optional display name from header
        db: Database session

    Returns:
        UserTest object
    """
    if not x_user_id or not x_user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-User-ID header is required"
        )

    # URL decode the user_id to support Chinese characters
    user_id = unquote(x_user_id.strip())
    display_name = unquote(x_display_name.strip()) if x_display_name else user_id

    # Look up existing user
    user = db.query(UserTest).filter(UserTest.user_id == user_id).first()

    if user:
        # Update display_name if provided and different
        if x_display_name and user.display_name != display_name:
            user.display_name = display_name
            user.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(user)
            print(f"Updated test user display name: {user_id} -> {display_name}")
    else:
        # Create new user
        user = UserTest(
            user_id=user_id,
            display_name=display_name,
            role=UserRoleTest.USER,
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"Created new test user: {user_id}")

    return user


def get_current_test_user(
    user: UserTest = Depends(get_or_create_test_user)
) -> UserTest:
    """
    Get the current test user and verify they are active.

    Args:
        user: UserTest from get_or_create_test_user

    Returns:
        Active UserTest object

    Raises:
        HTTPException if user is inactive
    """
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    return user
