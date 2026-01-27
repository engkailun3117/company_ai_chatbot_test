from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional

from database import get_db, engine, Base
from models import User, ChatSession, ChatMessage, CompanyOnboarding, Product, ChatSessionStatus
from schemas import (
    UserResponse,
    ChatMessageCreate, ChatResponse, ChatSessionResponse, ChatMessageResponse,
    OnboardingDataResponse
)
from config import get_settings
from auth import get_current_active_user
from ai_chatbot_handler import AIChatbotHandler
from file_processor import FileProcessor

# Create database tables
Base.metadata.create_all(bind=engine)

# Debug: Print configuration on startup
settings = get_settings()
print("=" * 60)
print("🔧 Backend Configuration:")
print(f"   Database: {settings.database_url[:30]}...")
print(f"   API Host: {settings.api_host}")
print(f"   API Port: {settings.api_port}")
print(f"   External JWT Secret: {settings.external_jwt_secret[:20]}... (length: {len(settings.external_jwt_secret)})")
print("=" * 60)

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise AI Chatbot API",
    description="AI-powered chatbot for collecting company onboarding information via conversational interface",
    version="3.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Nuxt app URL
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
        "message": "Enterprise AI Chatbot API is running",
        "version": "3.0.0",
        "features": ["external_jwt_auth", "ai_chatbot", "data_collection", "session_management"]
    }


# ============== Authentication Endpoints ==============

@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current authenticated user information

    Requires: Valid JWT token from main system in Authorization header

    This endpoint automatically syncs user data from the JWT token to the local database.
    If the user doesn't exist locally, it will be created automatically.
    """
    return current_user.to_dict()


# ============== Chatbot Endpoints ==============

@app.post("/api/chatbot/message", response_model=ChatResponse)
async def send_chatbot_message(
    chat_data: ChatMessageCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Send a message to the onboarding chatbot

    - **message**: User's message to the chatbot
    - **session_id**: Optional session ID to continue an existing conversation

    Requires: Authentication
    Returns: Chatbot response with session information
    """
    try:
        # Choose handler based on configuration
        settings = get_settings()

        handler = AIChatbotHandler(db, current_user.id, chat_data.session_id)

        # Create new session if needed
        if not handler.session:
            session = handler.create_session()
            # Send welcome message

            welcome_message = (
                "您好！我是企業導入 AI 助理 🤖\n\n"
                "我將用智能對話的方式協助您建立公司資料。您可以用自然的方式告訴我：\n"
                "• 產業別\n"
                "• 資本總額與專利數量\n"
                "• 認證資料（包括ESG認證）\n"
                "• 產品資訊\n\n"
                "您可以一次提供多個資訊，我會自動理解並記錄。\n"
                "讓我們開始吧！請告訴我您的公司資料。"
            )

            handler.add_message("assistant", welcome_message)

            return ChatResponse(
                session_id=session.id,
                message=welcome_message,
                completed=False,
                progress=handler.get_progress()
            )

        # Save user message
        handler.add_message("user", chat_data.message)

        # Process message and get response
        bot_response, is_completed = handler.process_message(chat_data.message)

        # Save bot response
        handler.add_message("assistant", bot_response)

        return ChatResponse(
            session_id=handler.session.id,
            message=bot_response,
            completed=is_completed,
            progress=handler.get_progress()
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your message: {str(e)}"
        )


@app.post("/api/chatbot/upload-file")
async def upload_file_for_extraction(
    file: UploadFile = File(...),
    session_id: Optional[int] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Upload a file (PDF, DOCX, Image) and extract company information using AI

    - **file**: File to upload (PDF, DOCX, JPG, PNG, TXT)
    - **session_id**: Optional session ID to add extracted data to existing session

    Requires: Authentication
    Returns: Extracted text and AI-processed company information

    Supported file types:
    - PDF documents (.pdf)
    - Word documents (.docx)
    - Images (.jpg, .png) - with OCR or AI Vision
    - Text files (.txt)

    Maximum file size: 10MB
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
                detail=f"Unsupported file type: {content_type}. Supported: PDF, DOCX, JPG, PNG, TXT"
            )

        # Process file and extract text
        result = processor.process_file(file_content, file.filename, content_type)

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )

        extracted_text = result["extracted_text"]

        # Use AI to extract structured data from text
        settings = get_settings()

        # Initialize AI handler
        handler = AIChatbotHandler(db, current_user.id, session_id)

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
                    "content": f"從以下文件內容中提取公司資訊：\n\n{extracted_text[:4000]}"  # Limit to 4000 chars
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

            # Proactively ask for the next field
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
            detail=f"Error processing file: {str(e)}"
        )


@app.get("/api/chatbot/sessions", response_model=List[ChatSessionResponse])
async def get_user_chat_sessions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get all chat sessions for the current user

    Requires: Authentication
    Returns: List of user's chat sessions
    """
    sessions = db.query(ChatSession).filter(
        ChatSession.user_id == current_user.id
    ).order_by(ChatSession.created_at.desc()).all()

    return [session.to_dict() for session in sessions]


@app.get("/api/chatbot/sessions/latest")
async def get_latest_active_session(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get the latest active chat session for the current user

    This endpoint helps avoid creating duplicate sessions on page refresh.
    It returns the most recent active session if one exists.

    Requires: Authentication
    Returns: Latest active session or null if none exists
    """
    # Find the most recent active session
    latest_session = db.query(ChatSession).filter(
        ChatSession.user_id == current_user.id,
        ChatSession.status == ChatSessionStatus.ACTIVE
    ).order_by(ChatSession.created_at.desc()).first()

    if latest_session:
        return {
            "session_id": latest_session.id,
            "status": latest_session.status.value,
            "created_at": latest_session.created_at.isoformat() if latest_session.created_at else None
        }

    return {"session_id": None}


@app.post("/api/chatbot/sessions/new")
async def create_new_session_with_context(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new chat session and copy company info from the latest session

    This endpoint is called when user explicitly clicks "New Session".
    It intelligently copies the latest company information to avoid duplicate records,
    while allowing the user to update the information if needed.

    Requires: Authentication
    Returns: New session ID with pre-populated company info
    """
    # Find the current company data (marked as is_current=True)
    latest_company_data = db.query(CompanyOnboarding).filter(
        CompanyOnboarding.user_id == current_user.id,
        CompanyOnboarding.is_current == True
    ).first()

    handler = AIChatbotHandler(db, current_user.id, None)

    # Create new session
    new_session = handler.create_session()

    # If we found previous company data, copy it to the new session
    if latest_company_data:
        # Get the newly created onboarding data
        new_onboarding = db.query(CompanyOnboarding).filter(
            CompanyOnboarding.chat_session_id == new_session.id
        ).first()

        if new_onboarding:
            # Copy chatbot collected fields from latest session
            new_onboarding.industry = latest_company_data.industry
            new_onboarding.capital_amount = latest_company_data.capital_amount
            new_onboarding.invention_patent_count = latest_company_data.invention_patent_count
            new_onboarding.utility_patent_count = latest_company_data.utility_patent_count
            new_onboarding.certification_count = latest_company_data.certification_count
            new_onboarding.esg_certification_count = latest_company_data.esg_certification_count
            new_onboarding.esg_certification = latest_company_data.esg_certification

            db.commit()

            # Copy products
            old_products = db.query(Product).filter(
                Product.onboarding_id == latest_company_data.id
            ).all()

            for old_product in old_products:
                new_product = Product(
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

    # Send welcome message
    if handler:
        if latest_company_data and latest_company_data.industry:
            welcome_message = (
                f"您好！歡迎回來！🤖\n\n"
                f"我已經為您載入了上次的公司資料：\n"
                f"• 產業別：{latest_company_data.industry}\n"
                f"• 資本總額：{latest_company_data.capital_amount or '未填寫'}臺幣\n\n"
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
    else:
        if latest_company_data and latest_company_data.industry:
            welcome_message = (
                f"您好！歡迎回來！👋\n\n"
                f"我已經為您載入了上次的公司資料：\n"
                f"• 產業別：{latest_company_data.industry}\n"
                f"• 資本總額：{latest_company_data.capital_amount or '未填寫'}億\n\n"
                f"讓我們繼續吧！請問您的公司所屬產業別是什麼？（例如：食品業、鋼鐵業、電子業等）"
            )
        else:
            welcome_message = (
                "您好！我是企業導入助理 👋\n\n"
                "我將協助您逐步建立公司資料。\n\n"
                "讓我們開始吧！請問您的公司所屬產業別是什麼？\n"
                "（例如：食品業、鋼鐵業、電子業等）"
            )

    handler.add_message("assistant", welcome_message)

    return {
        "session_id": new_session.id,
        "message": welcome_message,
        "company_info_copied": latest_company_data is not None,
        "progress": handler.get_progress()
    }


@app.get("/api/chatbot/sessions/{session_id}/messages", response_model=List[ChatMessageResponse])
async def get_session_messages(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get all messages for a specific chat session

    - **session_id**: The ID of the chat session

    Requires: Authentication
    Authorization: Users can only view their own sessions
    """
    # Verify session belongs to user
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )

    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.created_at).all()

    return [msg.to_dict() for msg in messages]


@app.get("/api/chatbot/data/{session_id}", response_model=OnboardingDataResponse)
async def get_onboarding_data(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get the onboarding data collected in a chat session

    - **session_id**: The ID of the chat session

    Requires: Authentication
    Authorization: Users can only view their own data
    """
    # Verify session belongs to user
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )

    onboarding_data = db.query(CompanyOnboarding).filter(
        CompanyOnboarding.chat_session_id == session_id
    ).first()

    if not onboarding_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No onboarding data found for this session"
        )

    return onboarding_data.to_dict()



@app.get("/api/chatbot/export/{session_id}")
async def export_onboarding_data(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Export onboarding data in the specified JSON format

    - **session_id**: The ID of the chat session

    Requires: Authentication
    Authorization: Users can only export their own data
    Returns: Data in the Chinese field name format
    """
    # Verify session belongs to user
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )

    onboarding_data = db.query(CompanyOnboarding).filter(
        CompanyOnboarding.chat_session_id == session_id
    ).first()

    if not onboarding_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No onboarding data found for this session"
        )

    # Return data in export format
    return onboarding_data.to_export_format()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)