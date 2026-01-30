"""
Test Database Models - For Internal Testing Environment

These models mirror the production models but use separate tables
with '_test' suffix to avoid mixing test and production data.
"""

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Enum, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from database import Base


class UserRoleTest(str, enum.Enum):
    """User role enumeration for test"""
    USER = "user"
    ADMIN = "admin"


class ChatSessionStatusTest(str, enum.Enum):
    """Chat session status enumeration for test"""
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class OnboardingStageTest(str, enum.Enum):
    """
    State machine for tracking which field is currently being collected.
    This ensures server-driven collection flow instead of relying on LLM.
    """
    INDUSTRY = "industry"
    CAPITAL_AMOUNT = "capital_amount"
    INVENTION_PATENT_COUNT = "invention_patent_count"
    UTILITY_PATENT_COUNT = "utility_patent_count"
    CERTIFICATION_COUNT = "certification_count"
    ESG_CERTIFICATION = "esg_certification"
    PRODUCT = "product"
    COMPLETED = "completed"


class ProductFieldTest(str, enum.Enum):
    """
    State machine for tracking which product field is currently being collected.
    Products require all 6 fields before being saved.
    """
    PRODUCT_ID = "product_id"
    PRODUCT_NAME = "product_name"
    PRICE = "price"
    MAIN_RAW_MATERIALS = "main_raw_materials"
    PRODUCT_STANDARD = "product_standard"
    TECHNICAL_ADVANTAGES = "technical_advantages"


class UserTest(Base):
    """Test User table - simplified authentication via user_id input"""

    __tablename__ = "users_test"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), unique=True, nullable=False, index=True)  # User-entered ID
    display_name = Column(String(100), nullable=True)  # Optional display name
    role = Column(Enum(UserRoleTest, native_enum=True, create_constraint=True, name='userrole_test'),
                  default=UserRoleTest.USER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "display_name": self.display_name,
            "role": self.role.value,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class ChatSessionTest(Base):
    """Test Chat session table for managing user chatbot conversations"""

    __tablename__ = "chat_sessions_test"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users_test.id"), nullable=False, index=True)
    status = Column(Enum(ChatSessionStatusTest, native_enum=True, create_constraint=True, name='chatsessionstatus_test'),
                    default=ChatSessionStatusTest.ACTIVE, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("UserTest")
    messages = relationship("ChatMessageTest", back_populates="session", cascade="all, delete-orphan")
    onboarding_data = relationship("CompanyOnboardingTest", back_populates="chat_session", uselist=False)

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class ChatMessageTest(Base):
    """Test Chat message table for storing conversation history"""

    __tablename__ = "chat_messages_test"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions_test.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    session = relationship("ChatSessionTest", back_populates="messages")

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class CompanyOnboardingTest(Base):
    """Test Company onboarding data collected through chatbot"""

    __tablename__ = "company_onboarding_test"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(Integer, ForeignKey("chat_sessions_test.id"), nullable=False, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users_test.id"), nullable=False, index=True)

    # Chatbot Data Collection
    industry = Column(String(100), nullable=True)  # 產業別
    capital_amount = Column(Integer, nullable=True)  # 資本總額（以臺幣為單位）
    invention_patent_count = Column(Integer, nullable=True)  # 發明專利數量
    utility_patent_count = Column(Integer, nullable=True)  # 新型專利數量
    certification_count = Column(Integer, nullable=True)  # 公司認證資料數量
    esg_certification_count = Column(Integer, nullable=True)  # ESG相關認證資料數量
    esg_certification = Column(Text, nullable=True)  # ESG相關認證資料

    # State Machine Fields - Server-driven collection flow
    # Note: values_callable ensures we use lowercase enum values to match PostgreSQL enum
    current_stage = Column(
        Enum(
            OnboardingStageTest,
            native_enum=True,
            create_constraint=True,
            name='onboardingstage_test',
            values_callable=lambda x: [e.value for e in x]
        ),
        default=OnboardingStageTest.INDUSTRY,
        nullable=False
    )  # Current field being collected

    # Product Collection State
    current_product_field = Column(
        Enum(
            ProductFieldTest,
            native_enum=True,
            create_constraint=True,
            name='productfield_test',
            values_callable=lambda x: [e.value for e in x]
        ),
        nullable=True
    )  # Current product field being collected (when in PRODUCT stage)
    current_product_draft = Column(Text, nullable=True)  # JSON string storing partial product data

    is_current = Column(Boolean, default=True, nullable=False, index=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("UserTest")
    chat_session = relationship("ChatSessionTest", back_populates="onboarding_data")
    products = relationship("ProductTest", back_populates="company_onboarding", cascade="all, delete-orphan")

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "id": self.id,
            "chat_session_id": self.chat_session_id,
            "user_id": self.user_id,
            "industry": self.industry,
            "capital_amount": self.capital_amount,
            "invention_patent_count": self.invention_patent_count,
            "utility_patent_count": self.utility_patent_count,
            "certification_count": self.certification_count,
            "esg_certification_count": self.esg_certification_count,
            "esg_certification": self.esg_certification,
            "current_stage": self.current_stage.value if self.current_stage else None,
            "current_product_field": self.current_product_field.value if self.current_product_field else None,
            "current_product_draft": self.current_product_draft,
            "is_current": self.is_current,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "products": [p.to_dict() for p in self.products] if self.products else []
        }

    def to_export_format(self):
        """Convert to the export JSON format"""
        return {
            "產業別": self.industry,
            "資本總額（以臺幣為單位）": self.capital_amount,
            "發明專利數量": self.invention_patent_count,
            "新型專利數量": self.utility_patent_count,
            "公司認證資料數量": self.certification_count,
            "ESG相關認證資料數量": self.esg_certification_count,
            "ESG相關認證資料": self.esg_certification,
            "產品": [p.to_export_format() for p in self.products] if self.products else []
        }


class ProductTest(Base):
    """Test Product information table"""

    __tablename__ = "products_test"

    id = Column(Integer, primary_key=True, index=True)
    onboarding_id = Column(Integer, ForeignKey("company_onboarding_test.id"), nullable=False, index=True)

    # Product Information
    product_id = Column(String(100), nullable=True, index=True)  # 產品ID
    product_name = Column(String(200), nullable=True)  # 產品名稱
    price = Column(String(50), nullable=True)  # 價格
    main_raw_materials = Column(String(500), nullable=True)  # 主要原料
    product_standard = Column(String(200), nullable=True)  # 產品規格
    technical_advantages = Column(Text, nullable=True)  # 技術優勢

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    company_onboarding = relationship("CompanyOnboardingTest", back_populates="products")

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "id": self.id,
            "onboarding_id": self.onboarding_id,
            "product_id": self.product_id,
            "product_name": self.product_name,
            "price": self.price,
            "main_raw_materials": self.main_raw_materials,
            "product_standard": self.product_standard,
            "technical_advantages": self.technical_advantages,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

    def to_export_format(self):
        """Convert to the export JSON format"""
        return {
            "產品ID": self.product_id,
            "產品名稱": self.product_name,
            "價格": self.price,
            "主要原料": self.main_raw_materials,
            "產品規格(尺寸、精度)": self.product_standard,
            "技術優勢": self.technical_advantages
        }
