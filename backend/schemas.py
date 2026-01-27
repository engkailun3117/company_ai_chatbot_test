from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ============== Authentication Schemas ==============

class UserResponse(BaseModel):
    """Schema for user response (synced from external JWT)"""
    id: int
    external_user_id: str
    username: str
    role: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============== Chatbot Schemas ==============

class ChatMessageCreate(BaseModel):
    """Schema for creating a chat message"""
    message: str = Field(..., min_length=1, description="User's message")
    session_id: Optional[int] = Field(None, description="Session ID (if continuing an existing session)")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "我想要開始設定我的公司資料",
                "session_id": None
            }
        }


class ChatMessageResponse(BaseModel):
    """Schema for chat message response"""
    id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    """Schema for chatbot response"""
    session_id: int
    message: str
    completed: bool = False
    progress: Optional[dict] = None

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": 1,
                "message": "您好！我是企業導入助理。我將協助您建立公司資料。首先，請問您的公司名稱是什麼？",
                "completed": False,
                "progress": {
                    "company_info_complete": False,
                    "products_complete": False
                }
            }
        }


class ChatSessionResponse(BaseModel):
    """Schema for chat session response"""
    id: int
    user_id: int
    status: str
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class ProductData(BaseModel):
    """Schema for product data"""
    product_id: Optional[str] = None
    product_name: Optional[str] = None
    price: Optional[str] = None
    main_raw_materials: Optional[str] = None
    product_standard: Optional[str] = None
    technical_advantages: Optional[str] = None


class OnboardingDataResponse(BaseModel):
    """Schema for onboarding data response"""
    id: int
    chat_session_id: int
    user_id: int
    industry: Optional[str]
    capital_amount: Optional[int]
    invention_patent_count: Optional[int]
    utility_patent_count: Optional[int]
    certification_count: Optional[int]
    esg_certification_count: Optional[int]
    esg_certification: Optional[str]
    is_current: bool
    created_at: datetime
    updated_at: datetime
    products: list[dict]

    class Config:
        from_attributes = True

