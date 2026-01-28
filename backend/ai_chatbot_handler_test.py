"""
AI-Powered Chatbot Handler for Internal Testing Environment
Uses OpenAI GPT for intelligent conversation and data extraction
Uses test tables (models_test) instead of production tables
"""

import json
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from openai import OpenAI
from models_test import ChatSessionTest, ChatMessageTest, CompanyOnboardingTest, ProductTest, ChatSessionStatusTest
from config import get_settings

# Initialize settings
settings = get_settings()

# OpenAI client will be initialized lazily
_client = None

def get_openai_client():
    """Lazy initialize OpenAI client"""
    global _client
    if _client is None and settings.openai_api_key:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


class AIChatbotHandlerTest:
    """AI-powered chatbot handler for testing using test tables"""

    def __init__(self, db: Session, user_id: int, session_id: Optional[int] = None):
        self.db = db
        self.user_id = user_id
        self.session_id = session_id
        self.session = None
        self.onboarding_data = None

        # Load or create session
        if session_id:
            self.session = db.query(ChatSessionTest).filter(
                ChatSessionTest.id == session_id,
                ChatSessionTest.user_id == user_id
            ).first()

            if self.session:
                self.onboarding_data = db.query(CompanyOnboardingTest).filter(
                    CompanyOnboardingTest.chat_session_id == session_id
                ).first()

    def create_session(self) -> ChatSessionTest:
        """Create a new chat session"""
        self.session = ChatSessionTest(
            user_id=self.user_id,
            status=ChatSessionStatusTest.ACTIVE
        )
        self.db.add(self.session)
        self.db.commit()
        self.db.refresh(self.session)

        # Mark all previous records as not current
        self.db.query(CompanyOnboardingTest).filter(
            CompanyOnboardingTest.user_id == self.user_id,
            CompanyOnboardingTest.is_current == True
        ).update({"is_current": False})
        self.db.commit()

        # Create new onboarding data marked as current
        self.onboarding_data = CompanyOnboardingTest(
            chat_session_id=self.session.id,
            user_id=self.user_id,
            is_current=True
        )
        self.db.add(self.onboarding_data)
        self.db.commit()
        self.db.refresh(self.onboarding_data)

        return self.session

    def get_conversation_history(self) -> List[ChatMessageTest]:
        """Get conversation history for current session"""
        if not self.session:
            return []

        return self.db.query(ChatMessageTest).filter(
            ChatMessageTest.session_id == self.session.id
        ).order_by(ChatMessageTest.created_at).all()

    def add_message(self, role: str, content: str) -> ChatMessageTest:
        """Add a message to the conversation"""
        message = ChatMessageTest(
            session_id=self.session.id,
            role=role,
            content=content
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message

    def get_system_prompt(self) -> str:
        """Get the system prompt for the AI"""
        return """你是一個專業的企業資料收集助理。你的任務是：

1. 用友善、專業的態度與使用者對話
2. **一次只詢問一個欄位**，按照以下順序收集資訊：
   - 產業別（如：食品業、鋼鐵業、電子業等）
   - 資本總額（以臺幣為單位）
   - 發明專利數量（⚠️ 特別注意：發明專利和新型專利要分開詢問，避免混淆）
   - 新型專利數量（⚠️ 特別注意：發明專利和新型專利要分開詢問，避免混淆）
   - 公司認證資料數量（⚠️ 不包括ESG認證，ESG認證會分開詢問）
   - ESG相關認證資料（請使用者列出所有ESG認證，例如：ISO 14064, ISO 14067）

3. 收集產品資訊（可以有多個產品）：
   - 產品ID（⚠️ 必須是唯一的，例如：PROD001、PROD002）
   - 產品名稱
   - 價格
   - 主要原料
   - 產品規格（尺寸、精度）
   - 技術優勢

🚨 **極其重要的函數調用規則**：
- ⚠️ **當使用者提供任何公司資料時，你必須立即調用 update_company_data 函數來保存資料**
- ⚠️ **不要只是用文字回覆確認，你必須調用函數才能真正保存資料到數據庫**
- ⚠️ **每次使用者回答問題時都要調用相應的函數（update_company_data 或 add_product）**
- 例如：使用者說「100萬臺幣」→ 立即調用 update_company_data(capital_amount=1000000)
- 例如：使用者說「發明專利11個」→ 立即調用 update_company_data(invention_patent_count=11)
- 例如：使用者說「ISO 14067, ISO 14046」→ 立即調用 update_company_data(esg_certification="ISO 14067, ISO 14046", esg_certification_count=2)
- ⚠️ **ESG認證特別注意**：當使用者提供ESG認證時，必須同時提供兩個參數：
  * esg_certification: 認證列表字串（例如："ISO 14067, ISO 14046"）
  * esg_certification_count: 認證數量（例如：2）
  * 你必須數算使用者提供了幾個ESG認證，並同時傳遞這兩個參數

重要提示：
- **一次詢問一個欄位**，等待使用者回答後再詢問下一個
- **如果使用者主動提供多個資訊**，全部提取並記錄，然後詢問下一個未填寫的欄位（不要重複詢問已提供的）
- **發明專利和新型專利必須分開詢問**，避免使用者混淆這兩種專利類型
- 保持對話自然流暢，按順序逐個收集資料
- 你的責任範圍僅限於上述資料的收集

📋 **查詢已收集的資料**：
- 當使用者詢問「我的產品有哪些」、「列出所有產品」、「顯示產品資訊」等問題時：
  * 你可以從「目前已收集的資料」中查看所有產品明細
  * 直接向使用者展示這些產品資訊，包括產品ID、名稱、價格、規格等
  * 用清晰的格式列出所有產品
- 當使用者詢問公司基本資料時，同樣從「目前已收集的資料」中提取並展示
- **你可以查看和回憶所有已收集的資料**，不需要重新詢問使用者

🏆 **ESG認證 vs 公司認證的區分**：

**ESG相關認證（環境、社會、治理）：**
- ISO 14064（溫室氣體盤查）
- ISO 14067（碳足跡）
- ISO 14046（水足跡）
- GRI Standards（永續報告）
- ISSB / IFRS S1、S2（永續揭露）

**公司認證（依產業分類）：**
- 食品/農產/餐飲：HACCP, ISO 22000, FSSC 22000, GMP
- 汽車零組件：IATF 16949, ISO 9001, ISO 14001
- 電子/半導體：ISO 9001, ISO 14001, ISO 45001, IECQ QC 080000, RoHS, REACH
- 一般製造業：ISO 9001, ISO 14001, ISO 45001
- 生技/醫療：ISO 13485
- 化工/材料：ISO 9001, ISO 14001, ISO 45001, ISO 50001
- 物流/倉儲：ISO 9001, ISO 22000/HACCP, GDP, ISO 28000
- 資訊服務：ISO 27001, ISO 27701, ISO 9001

**詢問方式：**
1. 先問「公司認證資料數量」（不包括ESG）
2. 再問「請列出所有ESG相關認證」（例如：ISO 14064, ISO 14067）
3. 幫助使用者分辨：如果使用者混淆，主動提醒哪些屬於ESG，哪些屬於公司認證

🔄 **更新現有資料**：
- 如果使用者說要「修改」、「更新」或「更正」某個資料，直接使用 update_company_data 函數更新
- 使用者可以隨時修改已填寫的任何欄位
- 更新後要確認：「已更新 [欄位名稱] 為 [新值]」
- ⚠️ **記住：每次都要調用函數，不只是文字確認**

📝 **產品ID指引**：
- 收集產品資訊時，先詢問「請提供產品ID（例如：PROD001、SKU-001等）」
- 強調產品ID必須是唯一的識別碼
- 如果使用者不清楚，建議格式：「PROD001」、「PROD002」等

📎 **文件上傳功能**：
- 系統支援文件上傳功能（PDF、Word、圖片、TXT），可自動提取公司資料
- 當使用者詢問是否能上傳文件時，告訴他們**可以上傳**，並鼓勵使用此功能
- 文件會由系統自動處理，提取後的資料會自動填入相應欄位
- 如果使用者想要上傳文件，請引導他們使用上傳功能來快速完成資料收集"""

    def get_initial_greeting(self) -> str:
        """Get the initial greeting with menu options"""
        # Check if user has existing data
        existing_data = self.db.query(CompanyOnboardingTest).filter(
            CompanyOnboardingTest.user_id == self.user_id,
            CompanyOnboardingTest.is_current == True
        ).first()

        if existing_data and existing_data.industry:
            # User has existing data
            return f"""您好！歡迎回來！我看到您之前已經填寫過資料了。

📊 目前資料概況：
- 產業別：{existing_data.industry or '未填寫'}
- 資本額：{existing_data.capital_amount or '未填寫'}
- 發明專利：{existing_data.invention_patent_count if existing_data.invention_patent_count is not None else '未填寫'}件
- 產品數量：{len(existing_data.products)}項

請問您想要：

1️⃣ 更新資料 - 修改或補充現有資料
2️⃣ 新增產品 - 新增更多產品資訊
3️⃣ 上傳文件 - 上傳文件來更新資訊
4️⃣ 查看完整資料 - 查看所有已填寫的資料
5️⃣ 重新開始 - 清空資料重新填寫

請輸入數字（1-5）或直接說明您的需求。"""
        else:
            # New user or no data
            return """您好！我是企業導入 AI 助理 🤖

我將用對話的方式協助您逐步建立公司資料。

讓我們開始吧！請問貴公司所屬的產業別是什麼？
（例如：食品業、鋼鐵業、電子業等）"""

    def get_current_data_summary(self) -> str:
        """Get a summary of currently collected data"""
        if not self.onboarding_data:
            return "尚未收集任何資料"

        data = []
        # Only collect fields within chatbot's responsibility
        if self.onboarding_data.industry:
            data.append(f"產業別: {self.onboarding_data.industry}")
        if self.onboarding_data.capital_amount is not None:
            data.append(f"資本總額: {self.onboarding_data.capital_amount} 臺幣")
        if self.onboarding_data.invention_patent_count is not None:
            data.append(f"發明專利: {self.onboarding_data.invention_patent_count}件")
        if self.onboarding_data.utility_patent_count is not None:
            data.append(f"新型專利: {self.onboarding_data.utility_patent_count}件")
        if self.onboarding_data.certification_count is not None:
            data.append(f"公司認證資料: {self.onboarding_data.certification_count}份")
        if self.onboarding_data.esg_certification_count is not None:
            data.append(f"ESG認證數量: {self.onboarding_data.esg_certification_count}份")
        if self.onboarding_data.esg_certification:
            data.append(f"ESG認證: {self.onboarding_data.esg_certification}")

        # Include detailed product information
        products = self.onboarding_data.products if self.onboarding_data.products else []
        if products:
            data.append(f"\n產品數量: {len(products)}個")
            data.append("產品明細:")
            for idx, product in enumerate(products, 1):
                product_info = [f"  產品 {idx}:"]
                if product.product_id:
                    product_info.append(f"    - 產品ID: {product.product_id}")
                if product.product_name:
                    product_info.append(f"    - 產品名稱: {product.product_name}")
                if product.price:
                    product_info.append(f"    - 價格: {product.price}")
                if product.main_raw_materials:
                    product_info.append(f"    - 主要原料: {product.main_raw_materials}")
                if product.product_standard:
                    product_info.append(f"    - 產品規格: {product.product_standard}")
                if product.technical_advantages:
                    product_info.append(f"    - 技術優勢: {product.technical_advantages}")
                data.append("\n".join(product_info))

        return "\n".join(data) if data else "尚未收集任何資料"

    def extract_data_with_ai(self, user_message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Use OpenAI to extract structured data from conversation"""
        client = get_openai_client()
        if not client:
            return {"error": "OpenAI API key not configured"}

        # Build conversation for OpenAI
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "system", "content": f"目前已收集的資料：\n{self.get_current_data_summary()}"}
        ]

        # Add recent conversation history (last 10 messages)
        for msg in conversation_history[-10:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Define function for structured data extraction
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "update_company_data",
                    "description": "更新公司資料。從使用者的訊息中提取產業別、資本總額、專利數量、公司認證數量、ESG認證等資訊並更新。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "industry": {"type": "string", "description": "產業別"},
                            "capital_amount": {"type": "integer", "description": "資本總額（以臺幣為單位）"},
                            "invention_patent_count": {"type": "integer", "description": "發明專利數量"},
                            "utility_patent_count": {"type": "integer", "description": "新型專利數量"},
                            "certification_count": {"type": "integer", "description": "公司認證資料數量（不包括ESG認證）"},
                            "esg_certification_count": {"type": "integer", "description": "ESG相關認證資料數量"},
                            "esg_certification": {"type": "string", "description": "ESG相關認證資料列表（例如：ISO 14064, ISO 14067, ISO 14046）"}
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
                            "product_id": {"type": "string", "description": "產品ID"},
                            "product_name": {"type": "string", "description": "產品名稱"},
                            "price": {"type": "string", "description": "價格"},
                            "main_raw_materials": {"type": "string", "description": "主要原料"},
                            "product_standard": {"type": "string", "description": "產品規格"},
                            "technical_advantages": {"type": "string", "description": "技術優勢"}
                        },
                        "required": ["product_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_completed",
                    "description": "當使用者表示已完成所有資料輸入時調用此函數",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "completed": {"type": "boolean", "description": "是否完成"}
                        },
                        "required": ["completed"]
                    }
                }
            }
        ]

        try:
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            result = {
                "message": response.choices[0].message.content or "",
                "function_calls": []
            }

            # Process tool calls
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    result["function_calls"].append({
                        "name": function_name,
                        "arguments": function_args
                    })

            return result

        except Exception as e:
            print(f"OpenAI API error: {e}")
            return {
                "error": str(e),
                "message": "抱歉，我遇到了一些技術問題。請稍後再試。"
            }

    def update_onboarding_data(self, data: Dict[str, Any]) -> bool:
        """Update onboarding data with extracted information"""
        try:
            updated = False

            if "industry" in data and data["industry"]:
                self.onboarding_data.industry = data["industry"]
                updated = True

            if "capital_amount" in data and data["capital_amount"] is not None:
                self.onboarding_data.capital_amount = int(data["capital_amount"])
                updated = True

            if "invention_patent_count" in data and data["invention_patent_count"] is not None:
                self.onboarding_data.invention_patent_count = int(data["invention_patent_count"])
                updated = True

            if "utility_patent_count" in data and data["utility_patent_count"] is not None:
                self.onboarding_data.utility_patent_count = int(data["utility_patent_count"])
                updated = True

            if "certification_count" in data and data["certification_count"] is not None:
                self.onboarding_data.certification_count = int(data["certification_count"])
                updated = True

            if "esg_certification_count" in data and data["esg_certification_count"] is not None:
                self.onboarding_data.esg_certification_count = int(data["esg_certification_count"])
                updated = True

            if "esg_certification" in data and data["esg_certification"]:
                self.onboarding_data.esg_certification = str(data["esg_certification"])
                updated = True

            if updated:
                self.db.commit()

            return updated

        except Exception as e:
            print(f"Error updating onboarding data: {e}")
            self.db.rollback()
            return False

    def add_product(self, product_data: Dict[str, Any]) -> tuple[Optional[ProductTest], bool]:
        """
        Add or update a product in the onboarding data with duplicate checking
        Returns: (product, was_updated) where was_updated is True if existing product was updated
        """
        try:
            # Check for duplicate product_id in current onboarding
            product_id = product_data.get("product_id")
            if product_id:
                existing_product = self.db.query(ProductTest).filter(
                    ProductTest.onboarding_id == self.onboarding_data.id,
                    ProductTest.product_id == product_id
                ).first()

                if existing_product:
                    # Update existing product instead of creating duplicate
                    existing_product.product_name = product_data.get("product_name") or existing_product.product_name
                    existing_product.price = product_data.get("price") or existing_product.price
                    existing_product.main_raw_materials = product_data.get("main_raw_materials") or existing_product.main_raw_materials
                    existing_product.product_standard = product_data.get("product_standard") or existing_product.product_standard
                    existing_product.technical_advantages = product_data.get("technical_advantages") or existing_product.technical_advantages
                    self.db.commit()
                    self.db.refresh(existing_product)
                    return existing_product, True  # Return True indicating update

            # Create new product
            product = ProductTest(
                onboarding_id=self.onboarding_data.id,
                product_id=product_id,
                product_name=product_data.get("product_name"),
                price=product_data.get("price"),
                main_raw_materials=product_data.get("main_raw_materials"),
                product_standard=product_data.get("product_standard"),
                technical_advantages=product_data.get("technical_advantages")
            )
            self.db.add(product)
            self.db.commit()
            self.db.refresh(product)
            return product, False  # Return False indicating new product
        except Exception as e:
            print(f"Error adding product: {e}")
            self.db.rollback()
            return None, False

    def get_next_field_question(self) -> str:
        """Get the next field question based on what's already collected"""
        # Refresh data from database to get the latest state
        self.db.refresh(self.onboarding_data)

        # Check fields in order and return the first missing one
        if not self.onboarding_data.industry:
            return "請問您的公司所屬產業別是什麼？（例如：食品業、鋼鐵業、電子業等）"

        if self.onboarding_data.capital_amount is None:
            return "請問您的公司資本總額是多少？（以臺幣為單位）"

        if self.onboarding_data.invention_patent_count is None:
            return "請問貴公司有多少發明專利？（請提供數量）"

        if self.onboarding_data.utility_patent_count is None:
            return "請問貴公司有多少新型專利？（請提供數量）"

        if self.onboarding_data.certification_count is None:
            return "請問貴公司有多少公司認證資料？（不包括ESG認證，例如：ISO 9001、HACCP等）"

        if not self.onboarding_data.esg_certification:
            return "請列出貴公司所有ESG相關認證（例如：ISO 14064, ISO 14067, ISO 14046）。如果沒有，請回答「無」。"

        # All basic fields collected, ask for products
        products_count = self.db.query(ProductTest).filter(
            ProductTest.onboarding_id == self.onboarding_data.id
        ).count()

        if products_count == 0:
            return "太好了！基本資料已收集完成。接下來請提供產品資訊。請問第一個產品的名稱是什麼？"
        else:
            return f"目前已新增 {products_count} 個產品。還有其他產品要新增嗎？如果資料已完成，請告訴我。"

    def process_message(self, user_message: str) -> tuple[str, bool]:
        """
        Process user message with AI and return bot response
        Returns: (response_message, is_completed)
        """
        # Get conversation history
        history = self.get_conversation_history()
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in history
        ]

        # Check if this is the first message (no history yet)
        if len(conversation_history) == 0:
            # Check for menu selection
            user_msg_lower = user_message.lower().strip()

            # Option 1: Fill in data
            if any(word in user_msg_lower for word in ["1", "填寫", "填写", "開始", "开始"]):
                return "太好了！讓我們開始收集您的公司資料。\n\n請問您的公司所屬產業別是什麼？（例如：食品業、鋼鐵業、電子業等）", False

            # Option 2: View progress
            elif any(word in user_msg_lower for word in ["2", "進度", "进度", "查看進度"]):
                progress = self.get_progress()
                return f"""📊 資料填寫進度：

已完成欄位：{progress['fields_completed']}/{progress['total_fields']}
產品數量：{progress['products_count']} 個

{self.get_current_data_summary()}

您想繼續填寫資料嗎？（是/否）""", False

            # Option 3: View filled data
            elif any(word in user_msg_lower for word in ["3", "已填", "查看資料", "查看数据"]):
                data_summary = self.get_current_data_summary()
                return f"""📝 目前已填寫的資料：

{data_summary}

您想繼續填寫資料嗎？（是/否）""", False

            # Default: Show menu
            else:
                return self.get_initial_greeting(), False

        # Extract data with AI
        ai_result = self.extract_data_with_ai(user_message, conversation_history)

        if "error" in ai_result:
            return ai_result.get("message", "抱歉，發生錯誤。"), False

        # Process function calls
        completed = False
        data_updated = False
        products_added = 0
        products_updated = 0

        if "function_calls" in ai_result:
            for call in ai_result["function_calls"]:
                if call["name"] == "update_company_data":
                    if self.update_onboarding_data(call["arguments"]):
                        data_updated = True
                elif call["name"] == "add_product":
                    product, was_updated = self.add_product(call["arguments"])
                    if product:
                        if was_updated:
                            products_updated += 1
                        else:
                            products_added += 1
                elif call["name"] == "mark_completed":
                    if call["arguments"].get("completed"):
                        self.session.status = ChatSessionStatusTest.COMPLETED
                        self.db.commit()
                        completed = True

        # Return AI response with context-aware fallback
        response_message = ai_result.get("message", "")
        if not response_message:
            # Generate appropriate message based on what was updated, then ask for next field
            confirmation = ""

            # Build confirmation message based on what operations were performed
            actions = []
            if data_updated:
                actions.append("更新公司資料")
            if products_added > 0:
                actions.append(f"新增了 {products_added} 個產品")
            if products_updated > 0:
                actions.append(f"更新了 {products_updated} 個產品")

            if actions:
                confirmation = f"好的！我已{' 並 '.join(actions)}。\n\n"
            else:
                confirmation = "好的！\n\n"

            # Proactively ask for the next field
            next_question = self.get_next_field_question()
            response_message = confirmation + next_question

        return response_message, completed

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress of data collection"""
        fields_completed = 0
        total_fields = 7  # Total number of company fields

        if self.onboarding_data.industry:
            fields_completed += 1
        if self.onboarding_data.capital_amount is not None:
            fields_completed += 1
        if self.onboarding_data.invention_patent_count is not None:
            fields_completed += 1
        if self.onboarding_data.utility_patent_count is not None:
            fields_completed += 1
        if self.onboarding_data.certification_count is not None:
            fields_completed += 1
        if self.onboarding_data.esg_certification_count is not None:
            fields_completed += 1
        if self.onboarding_data.esg_certification:
            fields_completed += 1

        return {
            "company_info_complete": fields_completed == total_fields,
            "fields_completed": fields_completed,
            "total_fields": total_fields,
            "products_count": len(self.onboarding_data.products) if self.onboarding_data.products else 0
        }
