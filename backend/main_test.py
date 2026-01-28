"""
Test API Server - For Internal Testing Environment

This module provides API endpoints for internal testing of the chatbot.
It uses simplified authentication (user_id header) and test database tables.
"""

from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional

from database import get_db, engine, Base
from models_test import (
    UserTest, ChatSessionTest, ChatMessageTest, CompanyOnboardingTest, ProductTest,
    ChatSessionStatusTest
)
from auth_test import get_current_test_user
from ai_chatbot_handler_test import AIChatbotHandlerTest
from file_processor import FileProcessor
from config import get_settings

# Create test database tables
Base.metadata.create_all(bind=engine)

# Debug: Print configuration on startup
settings = get_settings()
print("=" * 60)
print("🧪 TEST Backend Configuration:")
print(f"   Database: {settings.database_url[:30]}...")
print(f"   API Host: {settings.api_host}")
print(f"   API Port: {settings.api_port}")
print(f"   Mode: INTERNAL TESTING (Simplified Auth)")
print("=" * 60)

# Initialize FastAPI app
app = FastAPI(
    title="AI Chatbot Test API - 內部測試環境",
    description="內部測試用 AI Chatbot API - 簡化認證，使用獨立測試資料表",
    version="1.0.0-test"
)

# Configure CORS - allow all for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Health Check ==============

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "AI Chatbot Test API 正在運行",
        "version": "1.0.0-test",
        "mode": "internal_testing",
        "features": ["simplified_auth", "test_tables", "ai_chatbot"]
    }


# ============== User Endpoints ==============

@app.get("/api/test/user/me")
async def get_current_user_info(
    current_user: UserTest = Depends(get_current_test_user)
):
    """
    取得目前使用者資訊

    需要: X-User-ID header
    """
    return current_user.to_dict()


# ============== Chatbot Endpoints ==============

@app.post("/api/test/chatbot/message")
async def send_chatbot_message(
    message: str,
    session_id: Optional[int] = None,
    current_user: UserTest = Depends(get_current_test_user),
    db: Session = Depends(get_db)
):
    """
    傳送訊息給聊天機器人

    - **message**: 使用者訊息
    - **session_id**: 可選的 session ID（繼續現有對話）

    需要: X-User-ID header
    """
    try:
        handler = AIChatbotHandlerTest(db, current_user.id, session_id)

        # Create new session if needed
        if not handler.session:
            session = handler.create_session()

            welcome_message = (
                "您好！我是企業導入 AI 助理 🤖\n\n"
                "我將用對話的方式協助您逐步建立公司資料。您可以用自然的方式告訴我：\n"
                "• 產業別\n"
                "• 資本總額與專利數量\n"
                "• 認證資料（包括ESG認證）\n"
                "• 產品資訊\n\n"
                "您可以一次提供多個資訊，我會自動理解並記錄。\n"
                "讓我們開始吧！請告訴我您的公司資料。"
            )

            handler.add_message("assistant", welcome_message)

            return {
                "session_id": session.id,
                "message": welcome_message,
                "completed": False,
                "progress": handler.get_progress()
            }

        # Save user message
        handler.add_message("user", message)

        # Process message and get response
        bot_response, is_completed = handler.process_message(message)

        # Save bot response
        handler.add_message("assistant", bot_response)

        return {
            "session_id": handler.session.id,
            "message": bot_response,
            "completed": is_completed,
            "progress": handler.get_progress()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"處理訊息時發生錯誤: {str(e)}"
        )


@app.post("/api/test/chatbot/upload-file")
async def upload_file_for_extraction(
    file: UploadFile = File(...),
    session_id: Optional[int] = Form(None),
    current_user: UserTest = Depends(get_current_test_user),
    db: Session = Depends(get_db)
):
    """
    上傳文件並提取公司資訊

    - **file**: 要上傳的文件 (PDF, DOCX, JPG, PNG, TXT)
    - **session_id**: 可選的 session ID

    需要: X-User-ID header
    支援格式: PDF, DOCX, JPG, PNG, TXT (最大 10MB)
    """
    try:
        # Read file content
        file_content = await file.read()

        # Initialize file processor
        processor = FileProcessor()

        # Check file type
        content_type = file.content_type
        if not processor.is_supported(content_type):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支援的檔案格式: {content_type}。支援格式: PDF, DOCX, JPG, PNG, TXT"
            )

        # Process file and extract text
        result = processor.process_file(file_content, file.filename, content_type)

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )

        extracted_text = result["extracted_text"]

        # Initialize AI handler
        handler = AIChatbotHandlerTest(db, current_user.id, session_id)

        # Create session if needed
        if not handler.session:
            handler.create_session()
            session_id = handler.session.id

        # Use AI to extract structured company information
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)

        ai_response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": """你是一個資料提取專家。從提供的文件內容中提取以下公司資訊（如果存在）：
                    - 產業別
                    - 資本總額（以臺幣為單位）
                    - 發明專利數量
                    - 新型專利數量
                    - 公司認證資料數量（不包括ESG認證）
                    - ESG相關認證資料數量
                    - ESG相關認證列表（例如：ISO 14064, ISO 14067, ISO 14046）
                    - 產品資訊（產品ID、名稱、價格、原料、規格、技術優勢）

                    重要：區分一般公司認證與ESG認證。ESG相關認證包括：
                    - ISO 14064 (溫室氣體量化)
                    - ISO 14067 (碳足跡)
                    - ISO 14046 (水足跡)
                    - GRI Standards (永續報告)
                    - ISSB / IFRS S1, S2 (永續揭露)

                    以友善的方式總結找到的資訊，並告訴使用者已自動填入這些資料。
                    如果某些資訊未找到，禮貌地告知使用者可以稍後補充。"""
                },
                {
                    "role": "user",
                    "content": f"從以下文件內容中提取公司資訊：\n\n{extracted_text[:4000]}"
                }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "update_company_data",
                        "description": "更新公司資料",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "industry": {"type": "string"},
                                "capital_amount": {"type": "integer"},
                                "invention_patent_count": {"type": "integer"},
                                "utility_patent_count": {"type": "integer"},
                                "certification_count": {"type": "integer"},
                                "esg_certification_count": {"type": "integer"},
                                "esg_certification": {"type": "string"}
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "add_product",
                        "description": "新增產品資訊",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "product_id": {"type": "string"},
                                "product_name": {"type": "string"},
                                "price": {"type": "string"},
                                "main_raw_materials": {"type": "string"},
                                "product_standard": {"type": "string"},
                                "technical_advantages": {"type": "string"}
                            },
                            "required": ["product_name"]
                        }
                    }
                }
            ],
            tool_choice="auto"
        )

        # Process AI response and update database
        ai_message = ai_response.choices[0].message.content or ""
        data_updated = False
        products_added = 0

        if ai_response.choices[0].message.tool_calls:
            import json
            for tool_call in ai_response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if function_name == "update_company_data":
                    if handler.update_onboarding_data(function_args):
                        data_updated = True
                elif function_name == "add_product":
                    if handler.add_product(function_args):
                        products_added += 1

        # Generate context-aware message if AI didn't provide one
        if not ai_message:
            confirmation = ""
            if data_updated and products_added > 0:
                confirmation = f"已從文件中提取公司資料並新增了 {products_added} 個產品！資料已自動填入對應欄位。\n\n"
            elif data_updated:
                confirmation = "已從文件中提取公司資料！資料已自動填入對應欄位。\n\n"
            elif products_added > 0:
                confirmation = f"已從文件中提取 {products_added} 個產品資訊！資料已自動填入。\n\n"
            else:
                confirmation = "已處理文件，但未找到可提取的公司資料。\n\n"

            next_question = handler.get_next_field_question()
            ai_message = confirmation + next_question

        # Save the AI message to conversation history
        handler.add_message("assistant", f"📄 已處理文件：{file.filename}\n\n{ai_message}")

        return {
            "success": True,
            "filename": file.filename,
            "session_id": session_id,
            "message": ai_message,
            "extracted_text_length": len(extracted_text),
            "data_updated": data_updated,
            "products_added": products_added,
            "progress": handler.get_progress()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"處理文件時發生錯誤: {str(e)}"
        )


@app.get("/api/test/chatbot/sessions")
async def get_user_chat_sessions(
    current_user: UserTest = Depends(get_current_test_user),
    db: Session = Depends(get_db)
):
    """
    取得使用者的所有對話 sessions

    需要: X-User-ID header
    """
    sessions = db.query(ChatSessionTest).filter(
        ChatSessionTest.user_id == current_user.id
    ).order_by(ChatSessionTest.created_at.desc()).all()

    return [session.to_dict() for session in sessions]


@app.get("/api/test/chatbot/sessions/latest")
async def get_latest_active_session(
    current_user: UserTest = Depends(get_current_test_user),
    db: Session = Depends(get_db)
):
    """
    取得使用者最新的 active session

    需要: X-User-ID header
    """
    latest_session = db.query(ChatSessionTest).filter(
        ChatSessionTest.user_id == current_user.id,
        ChatSessionTest.status == ChatSessionStatusTest.ACTIVE
    ).order_by(ChatSessionTest.created_at.desc()).first()

    if latest_session:
        return {
            "session_id": latest_session.id,
            "status": latest_session.status.value,
            "created_at": latest_session.created_at.isoformat() if latest_session.created_at else None
        }

    return {"session_id": None}


@app.post("/api/test/chatbot/sessions/new")
async def create_new_session_with_context(
    current_user: UserTest = Depends(get_current_test_user),
    db: Session = Depends(get_db)
):
    """
    建立新的對話 session（會複製之前的公司資料）

    需要: X-User-ID header
    """
    # Find the current company data
    latest_company_data = db.query(CompanyOnboardingTest).filter(
        CompanyOnboardingTest.user_id == current_user.id,
        CompanyOnboardingTest.is_current == True
    ).first()

    handler = AIChatbotHandlerTest(db, current_user.id, None)

    # Create new session
    new_session = handler.create_session()

    # Copy previous company data if exists
    if latest_company_data:
        new_onboarding = db.query(CompanyOnboardingTest).filter(
            CompanyOnboardingTest.chat_session_id == new_session.id
        ).first()

        if new_onboarding:
            new_onboarding.industry = latest_company_data.industry
            new_onboarding.capital_amount = latest_company_data.capital_amount
            new_onboarding.invention_patent_count = latest_company_data.invention_patent_count
            new_onboarding.utility_patent_count = latest_company_data.utility_patent_count
            new_onboarding.certification_count = latest_company_data.certification_count
            new_onboarding.esg_certification_count = latest_company_data.esg_certification_count
            new_onboarding.esg_certification = latest_company_data.esg_certification
            db.commit()

            # Copy products
            old_products = db.query(ProductTest).filter(
                ProductTest.onboarding_id == latest_company_data.id
            ).all()

            for old_product in old_products:
                new_product = ProductTest(
                    onboarding_id=new_onboarding.id,
                    product_id=old_product.product_id,
                    product_name=old_product.product_name,
                    price=old_product.price,
                    main_raw_materials=old_product.main_raw_materials,
                    product_standard=old_product.product_standard,
                    technical_advantages=old_product.technical_advantages
                )
                db.add(new_product)
            db.commit()

    # Generate welcome message
    if latest_company_data and latest_company_data.industry:
        welcome_message = (
            f"您好！歡迎回來！🤖\n\n"
            f"我已經為您載入了上次的公司資料：\n"
            f"• 產業別：{latest_company_data.industry}\n"
            f"• 資本總額：{latest_company_data.capital_amount or '未填寫'} 臺幣\n\n"
            f"您可以告訴我需要更新哪些資訊，或是新增/修改產品資料。\n"
            f"如果資料都正確，您也可以直接確認完成。"
        )
    else:
        welcome_message = (
            "您好！我是企業導入 AI 助理 🤖\n\n"
            "我將用對話的方式協助您逐步建立公司資料。\n\n"
            "讓我們開始吧！請問貴公司所屬的產業別是什麼？\n"
            "（例如：食品業、鋼鐵業、電子業等）"
        )

    handler.add_message("assistant", welcome_message)

    return {
        "session_id": new_session.id,
        "message": welcome_message,
        "company_info_copied": latest_company_data is not None,
        "progress": handler.get_progress()
    }


@app.get("/api/test/chatbot/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: int,
    current_user: UserTest = Depends(get_current_test_user),
    db: Session = Depends(get_db)
):
    """
    取得特定 session 的所有訊息

    需要: X-User-ID header
    """
    session = db.query(ChatSessionTest).filter(
        ChatSessionTest.id == session_id,
        ChatSessionTest.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="找不到對話 session"
        )

    messages = db.query(ChatMessageTest).filter(
        ChatMessageTest.session_id == session_id
    ).order_by(ChatMessageTest.created_at).all()

    return [msg.to_dict() for msg in messages]


@app.get("/api/test/chatbot/data/{session_id}")
async def get_onboarding_data(
    session_id: int,
    current_user: UserTest = Depends(get_current_test_user),
    db: Session = Depends(get_db)
):
    """
    取得特定 session 收集的公司資料

    需要: X-User-ID header
    """
    session = db.query(ChatSessionTest).filter(
        ChatSessionTest.id == session_id,
        ChatSessionTest.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="找不到對話 session"
        )

    onboarding_data = db.query(CompanyOnboardingTest).filter(
        CompanyOnboardingTest.chat_session_id == session_id
    ).first()

    if not onboarding_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="此 session 尚無公司資料"
        )

    return onboarding_data.to_dict()


@app.get("/api/test/chatbot/export/{session_id}")
async def export_onboarding_data(
    session_id: int,
    current_user: UserTest = Depends(get_current_test_user),
    db: Session = Depends(get_db)
):
    """
    匯出公司資料（中文欄位名稱格式）

    需要: X-User-ID header
    """
    session = db.query(ChatSessionTest).filter(
        ChatSessionTest.id == session_id,
        ChatSessionTest.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="找不到對話 session"
        )

    onboarding_data = db.query(CompanyOnboardingTest).filter(
        CompanyOnboardingTest.chat_session_id == session_id
    ).first()

    if not onboarding_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="此 session 尚無公司資料"
        )

    return onboarding_data.to_export_format()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    # Run on port 8001 to avoid conflict with production API
    uvicorn.run(app, host=settings.api_host, port=8001)
