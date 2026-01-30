"""
AI-Powered Chatbot Handler for Company Onboarding Assistant
Uses OpenAI GPT for intelligent conversation and data extraction
"""

import json
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from openai import OpenAI
from models import (
    ChatSession, ChatMessage, CompanyOnboarding, Product, ChatSessionStatus,
    OnboardingStage, ProductField
)
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


class AIChatbotHandler:
    """AI-powered chatbot handler using OpenAI"""

    def __init__(self, db: Session, user_id: int, session_id: Optional[int] = None):
        self.db = db
        self.user_id = user_id
        self.session_id = session_id
        self.session = None
        self.onboarding_data = None

        # Load or create session
        if session_id:
            self.session = db.query(ChatSession).filter(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id
            ).first()

            if self.session:
                self.onboarding_data = db.query(CompanyOnboarding).filter(
                    CompanyOnboarding.chat_session_id == session_id
                ).first()

    def create_session(self) -> ChatSession:
        """Create a new chat session"""
        self.session = ChatSession(
            user_id=self.user_id,
            status=ChatSessionStatus.ACTIVE
        )
        self.db.add(self.session)
        self.db.commit()
        self.db.refresh(self.session)

        # Mark all previous records as not current
        self.db.query(CompanyOnboarding).filter(
            CompanyOnboarding.user_id == self.user_id,
            CompanyOnboarding.is_current == True
        ).update({"is_current": False})
        self.db.commit()

        # Create new onboarding data marked as current
        self.onboarding_data = CompanyOnboarding(
            chat_session_id=self.session.id,
            user_id=self.user_id,
            is_current=True
        )
        self.db.add(self.onboarding_data)
        self.db.commit()
        self.db.refresh(self.onboarding_data)

        return self.session

    # ================== STATE MACHINE METHODS ==================

    # Stage order for company data collection
    STAGE_ORDER = [
        OnboardingStage.INDUSTRY,
        OnboardingStage.CAPITAL_AMOUNT,
        OnboardingStage.INVENTION_PATENT_COUNT,
        OnboardingStage.UTILITY_PATENT_COUNT,
        OnboardingStage.CERTIFICATION_COUNT,
        OnboardingStage.ESG_CERTIFICATION,
        OnboardingStage.PRODUCT,
        OnboardingStage.COMPLETED,
    ]

    # Product field order
    PRODUCT_FIELD_ORDER = [
        ProductField.PRODUCT_ID,
        ProductField.PRODUCT_NAME,
        ProductField.PRICE,
        ProductField.MAIN_RAW_MATERIALS,
        ProductField.PRODUCT_STANDARD,
        ProductField.TECHNICAL_ADVANTAGES,
    ]

    # Stage to field mapping
    STAGE_TO_FIELD = {
        OnboardingStage.INDUSTRY: "industry",
        OnboardingStage.CAPITAL_AMOUNT: "capital_amount",
        OnboardingStage.INVENTION_PATENT_COUNT: "invention_patent_count",
        OnboardingStage.UTILITY_PATENT_COUNT: "utility_patent_count",
        OnboardingStage.CERTIFICATION_COUNT: "certification_count",
        OnboardingStage.ESG_CERTIFICATION: "esg_certification",
    }

    # Stage to display name mapping (Chinese)
    STAGE_TO_DISPLAY_NAME = {
        OnboardingStage.INDUSTRY: "產業別",
        OnboardingStage.CAPITAL_AMOUNT: "資本總額",
        OnboardingStage.INVENTION_PATENT_COUNT: "發明專利數量",
        OnboardingStage.UTILITY_PATENT_COUNT: "新型專利數量",
        OnboardingStage.CERTIFICATION_COUNT: "公司認證資料數量",
        OnboardingStage.ESG_CERTIFICATION: "ESG相關認證",
        OnboardingStage.PRODUCT: "產品資訊",
    }

    # Product field to display name mapping
    PRODUCT_FIELD_TO_DISPLAY_NAME = {
        ProductField.PRODUCT_ID: "產品ID",
        ProductField.PRODUCT_NAME: "產品名稱",
        ProductField.PRICE: "價格",
        ProductField.MAIN_RAW_MATERIALS: "主要原料",
        ProductField.PRODUCT_STANDARD: "產品規格",
        ProductField.TECHNICAL_ADVANTAGES: "技術優勢",
    }

    def get_current_stage(self) -> OnboardingStage:
        """Get current stage from onboarding data"""
        if not self.onboarding_data:
            return OnboardingStage.INDUSTRY
        return self.onboarding_data.current_stage or OnboardingStage.INDUSTRY

    def get_expected_field(self) -> Optional[str]:
        """Get the field name expected to be extracted at current stage"""
        stage = self.get_current_stage()
        if stage == OnboardingStage.PRODUCT:
            # Return current product field
            product_field = self.onboarding_data.current_product_field
            if product_field:
                return product_field.value
            return ProductField.PRODUCT_ID.value
        elif stage == OnboardingStage.COMPLETED:
            return None
        else:
            return self.STAGE_TO_FIELD.get(stage)

    def get_expected_tool(self) -> str:
        """Get the tool name expected to be called at current stage"""
        stage = self.get_current_stage()
        if stage == OnboardingStage.PRODUCT:
            return "collect_product_field"
        elif stage == OnboardingStage.COMPLETED:
            return "mark_completed"
        else:
            return "update_company_data"

    def advance_stage(self) -> OnboardingStage:
        """Move to the next stage in the state machine"""
        current_stage = self.get_current_stage()

        # Find next stage
        try:
            current_index = self.STAGE_ORDER.index(current_stage)
            if current_index < len(self.STAGE_ORDER) - 1:
                next_stage = self.STAGE_ORDER[current_index + 1]
                self.onboarding_data.current_stage = next_stage

                # If entering product stage, initialize product field
                if next_stage == OnboardingStage.PRODUCT:
                    self.onboarding_data.current_product_field = ProductField.PRODUCT_ID
                    self.onboarding_data.current_product_draft = json.dumps({})

                self.db.commit()
                return next_stage
        except ValueError:
            pass

        return current_stage

    def advance_product_field(self) -> Optional[ProductField]:
        """Move to the next product field, or return None if product is complete"""
        current_field = self.onboarding_data.current_product_field
        if not current_field:
            current_field = ProductField.PRODUCT_ID

        try:
            current_index = self.PRODUCT_FIELD_ORDER.index(current_field)
            if current_index < len(self.PRODUCT_FIELD_ORDER) - 1:
                next_field = self.PRODUCT_FIELD_ORDER[current_index + 1]
                self.onboarding_data.current_product_field = next_field
                self.db.commit()
                return next_field
            else:
                # Product collection is complete
                return None
        except ValueError:
            return None

    def reset_product_draft(self):
        """Reset product draft for collecting a new product"""
        self.onboarding_data.current_product_field = ProductField.PRODUCT_ID
        self.onboarding_data.current_product_draft = json.dumps({})
        self.db.commit()

    def get_product_draft(self) -> Dict[str, Any]:
        """Get current product draft as dict"""
        if not self.onboarding_data.current_product_draft:
            return {}
        try:
            return json.loads(self.onboarding_data.current_product_draft)
        except json.JSONDecodeError:
            return {}

    def update_product_draft(self, field: str, value: str) -> Dict[str, Any]:
        """Update a field in the product draft"""
        draft = self.get_product_draft()
        draft[field] = value
        self.onboarding_data.current_product_draft = json.dumps(draft)
        self.db.commit()
        return draft

    def is_product_draft_complete(self) -> bool:
        """Check if all required product fields are filled"""
        draft = self.get_product_draft()
        required_fields = [f.value for f in self.PRODUCT_FIELD_ORDER]
        return all(draft.get(f) for f in required_fields)

    def save_product_from_draft(self) -> Optional[Product]:
        """Save the completed product draft to database"""
        draft = self.get_product_draft()
        if not self.is_product_draft_complete():
            return None

        # Check for duplicate product_id
        product_id = draft.get("product_id")
        if product_id:
            existing_product = self.db.query(Product).filter(
                Product.onboarding_id == self.onboarding_data.id,
                Product.product_id == product_id
            ).first()

            if existing_product:
                # Update existing product
                existing_product.product_name = draft.get("product_name")
                existing_product.price = draft.get("price")
                existing_product.main_raw_materials = draft.get("main_raw_materials")
                existing_product.product_standard = draft.get("product_standard")
                existing_product.technical_advantages = draft.get("technical_advantages")
                self.db.commit()
                self.db.refresh(existing_product)
                self.reset_product_draft()
                return existing_product

        # Create new product
        product = Product(
            onboarding_id=self.onboarding_data.id,
            product_id=draft.get("product_id"),
            product_name=draft.get("product_name"),
            price=draft.get("price"),
            main_raw_materials=draft.get("main_raw_materials"),
            product_standard=draft.get("product_standard"),
            technical_advantages=draft.get("technical_advantages")
        )
        self.db.add(product)
        self.db.commit()
        self.db.refresh(product)
        self.reset_product_draft()
        return product

    def sync_stage_with_data(self):
        """
        Sync current_stage with actual data state.
        This is needed when data was imported or the stage got out of sync.
        """
        if not self.onboarding_data:
            return

        # Check each field and find the first empty one
        if not self.onboarding_data.industry:
            self.onboarding_data.current_stage = OnboardingStage.INDUSTRY
        elif self.onboarding_data.capital_amount is None:
            self.onboarding_data.current_stage = OnboardingStage.CAPITAL_AMOUNT
        elif self.onboarding_data.invention_patent_count is None:
            self.onboarding_data.current_stage = OnboardingStage.INVENTION_PATENT_COUNT
        elif self.onboarding_data.utility_patent_count is None:
            self.onboarding_data.current_stage = OnboardingStage.UTILITY_PATENT_COUNT
        elif self.onboarding_data.certification_count is None:
            self.onboarding_data.current_stage = OnboardingStage.CERTIFICATION_COUNT
        elif not self.onboarding_data.esg_certification:
            self.onboarding_data.current_stage = OnboardingStage.ESG_CERTIFICATION
        else:
            # All basic fields filled, move to product stage
            self.onboarding_data.current_stage = OnboardingStage.PRODUCT
            if not self.onboarding_data.current_product_field:
                self.onboarding_data.current_product_field = ProductField.PRODUCT_ID
                self.onboarding_data.current_product_draft = json.dumps({})

        self.db.commit()

    # ================== END STATE MACHINE METHODS ==================

    def get_conversation_history(self) -> List[ChatMessage]:
        """Get conversation history for current session"""
        if not self.session:
            return []

        return self.db.query(ChatMessage).filter(
            ChatMessage.session_id == self.session.id
        ).order_by(ChatMessage.created_at).all()

    def add_message(self, role: str, content: str) -> ChatMessage:
        """Add a message to the conversation"""
        message = ChatMessage(
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

📌 **核心原則：讓使用者感受到填寫資料的價值**

1. 用友善、專業的態度與使用者對話
2. **每次回覆都要顯示進度**，格式：【進度：X/6 已完成】（基本資料共6項，產品另計）
3. **適時提醒填寫資料的效益**：
   - 【推薦引擎】可幫助曝光產品、尋找合作夥伴
   - 【補助引擎】可協助申請政府補助案

4. **一次只詢問一個欄位**，按照以下順序收集資訊：
   - 產業別（如：食品業、鋼鐵業、電子業等）
   - 資本總額（以臺幣為單位）
   - 發明專利數量（⚠️ 特別注意：發明專利和新型專利要分開詢問，避免混淆）
   - 新型專利數量（⚠️ 特別注意：發明專利和新型專利要分開詢問，避免混淆）
   - 公司認證資料數量（⚠️ 不包括ESG認證，ESG認證會分開詢問）
   - ESG相關認證資料（請使用者列出所有ESG認證，例如：ISO 14064, ISO 14067）

5. 收集產品資訊（可以有多個產品）：
   ⚠️ **產品收集流程 - 必須逐一詢問每個欄位（共6項）**：
   a. 先問「產品ID」（唯一識別碼，例如：PROD001）→ 【產品進度：1/6】
   b. 再問「產品名稱」→ 【產品進度：2/6】
   c. 再問「價格」→ 【產品進度：3/6】
   d. 再問「主要原料」（若無請填「-」）→ 【產品進度：4/6】
   e. 再問「產品規格（尺寸、精度）」（若無請填「-」）→ 【產品進度：5/6】
   f. 最後問「技術優勢」（若無請填「-」）→ 【產品進度：6/6】
   g. 收集完所有6個欄位後，才調用 add_product 函數新增產品

   📊 **產品進度顯示**：每次詢問產品欄位時，要顯示【產品進度：X/6 已填寫】
   例如：「✅ 已記錄產品名稱。【產品進度：2/6 已填寫】\n接下來請提供**價格**」

   ⚠️ **重要**：不要只收到部分資訊就調用 add_product！
   - 必須收集完整的6個欄位才能新增產品
   - 如果使用者只提供部分資訊，要繼續詢問其他欄位
   - ⚠️ **在收集產品資訊期間，不要調用 update_company_data！**
   - 如果你剛問了「產品價格」，使用者回答「1000」，這是產品價格，不是公司資料！

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

⚠️ **產品收集期間的特別注意**：
- 如果基本資料已完成（6/6），且你正在收集產品資訊，使用者的回答應該被視為產品資料
- 例如：你問「產品價格」，使用者回「1000」→ 這是產品價格，不要調用 update_company_data
- 例如：你問「主要原料」，使用者回「矽晶圓」→ 這是產品原料，不要調用 update_company_data
- **只有在收集完產品的全部6個欄位後，才調用 add_product 函數**

重要提示：
- **每次回覆都顯示進度**：「【進度：X/6 已完成】」讓使用者知道還剩多少（基本資料共6項）
- **一次詢問一個欄位**，等待使用者回答後再詢問下一個
- **如果使用者主動提供多個資訊**，全部提取並記錄，然後詢問下一個未填寫的欄位（不要重複詢問已提供的）
- **發明專利和新型專利必須分開詢問**，避免使用者混淆這兩種專利類型
- 保持對話自然流暢，按順序逐個收集資料
- **適時鼓勵使用者**，例如：「太好了！資料越完整，推薦引擎越能精準為您配對！」
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
- 如果使用者想要上傳文件，請引導他們使用上傳功能來快速完成資料收集

🎯 **基本資料完成時的格式**：
當所有基本資料（6/6）收集完成時，必須按照以下格式回覆：

```
🎉 太好了！基本資料已收集完成 【進度：6/6 已完成】

══════════════════════════════
📋 基本資料摘要
══════════════════════════════
• 產業別：[產業別]
• 資本額：[資本總額] 臺幣
• 發明專利：[發明專利數量] 件
• 新型專利：[新型專利數量] 件
• 公司認證：[公司認證數量] 項
• ESG認證：[ESG認證]

接下來請提供產品資訊，讓【推薦引擎】能幫助您曝光產品。

我會逐一詢問每個產品的詳細資訊（共6項）：
• 產品ID → 產品名稱 → 價格 → 主要原料 → 規格 → 技術優勢
（如果有多個產品，建議直接跟著格式上傳檔案）

請先提供第一個產品的**產品ID**（例如：PROD001）
【產品進度：0/6 已填寫】
```

⚠️ **重要**：你必須從「目前已收集的資料」中提取真實的值來顯示，不要使用佔位符

🚫 **重要：何時才能調用 mark_completed**：
- ⚠️ 基本資料（6項）填完後，**不要**調用 mark_completed
- ⚠️ 基本資料填完後要繼續收集產品資訊
- ✅ 只有當使用者明確說「完成」、「結束」、「不用了」、「沒有其他產品」時才調用 mark_completed
- ✅ 如果使用者還沒提供任何產品，要先詢問是否要新增產品
- 如果使用者尚未填寫產品資訊，提醒他們「新增產品資訊可讓推薦引擎更精準為您配對商機」

📊 **進度回報範例**：
【基本資料進度】
- 使用者回答第1題後：「✅ 已記錄產業別！【進度：1/6 已完成，還剩 5 項】」
- 使用者回答第4題後：「✅ 很好！【進度：4/6 已完成】再 2 項就完成基本資料了！」
- 完成所有基本資料後：「🎉【進度：6/6 已完成】太棒了！基本資料收集完畢！接下來您可以新增產品資訊」

【產品進度】
- 收到產品ID後：「✅ 已記錄產品ID。【產品進度：1/6 已填寫】\n接下來請提供**產品名稱**」
- 收到產品名稱後：「✅ 已記錄產品名稱。【產品進度：2/6 已填寫】\n接下來請提供**價格**」
- 收到價格後：「✅ 已記錄價格。【產品進度：3/6 已填寫】\n接下來請提供**主要原料**」
- 收到主要原料後：「✅ 已記錄主要原料。【產品進度：4/6 已填寫】\n接下來請提供**產品規格**」
- 收到產品規格後：「✅ 已記錄產品規格。【產品進度：5/6 已填寫】\n接下來請提供**技術優勢**」
- 收到技術優勢後（產品完成）：顯示產品已新增 + 所有產品摘要列表"""

    def get_missing_fields(self) -> list:
        """Get list of missing fields"""
        missing = []
        if not self.onboarding_data.industry:
            missing.append("產業別")
        if self.onboarding_data.capital_amount is None:
            missing.append("資本總額")
        if self.onboarding_data.invention_patent_count is None:
            missing.append("發明專利數量")
        if self.onboarding_data.utility_patent_count is None:
            missing.append("新型專利數量")
        if self.onboarding_data.certification_count is None:
            missing.append("公司認證資料")
        # ESG counts as ONE field
        if not self.onboarding_data.esg_certification:
            missing.append("ESG相關認證")
        return missing

    def get_progress_string(self) -> str:
        """Get formatted progress string"""
        progress = self.get_progress()
        fields_done = progress['fields_completed']
        total = progress['total_fields']
        remaining = total - fields_done
        return f"【進度：{fields_done}/{total} 已完成，還剩 {remaining} 項】"

    def get_company_summary(self) -> str:
        """Get a formatted summary of company basic info"""
        if not self.onboarding_data:
            return "尚未有公司基本資料。"

        data = self.onboarding_data
        summary = "\n══════════════════════════════\n🏢 公司基本資料：\n══════════════════════════════\n"
        summary += f"  • 產業別：{data.industry or '未填寫'}\n"
        summary += f"  • 資本總額：{data.capital_amount or '未填寫'}\n"
        summary += f"  • 發明專利數量：{data.invention_patent_count if data.invention_patent_count is not None else '未填寫'}\n"
        summary += f"  • 新型專利數量：{data.utility_patent_count if data.utility_patent_count is not None else '未填寫'}\n"
        summary += f"  • 公司認證資料數量：{data.certification_count if data.certification_count is not None else '未填寫'}\n"
        summary += f"  • ESG相關認證數量：{data.esg_certification_count if data.esg_certification_count is not None else '未填寫'}\n"
        summary += f"  • ESG相關認證資料：{data.esg_certification or '未填寫'}\n"

        return summary

    def get_products_summary(self) -> str:
        """Get a formatted summary of all products"""
        if not self.onboarding_data or not self.onboarding_data.products:
            return ""

        products = self.onboarding_data.products
        if not products:
            return ""

        summary = f"\n══════════════════════════════\n📋 已記錄的產品列表（共 {len(products)} 個）：\n══════════════════════════════\n"
        for idx, product in enumerate(products, 1):
            summary += f"\n**產品 {idx}**：{product.product_name or '未命名'}\n"
            summary += f"  • 產品ID：{product.product_id or '-'}\n"
            summary += f"  • 價格：{product.price or '-'}\n"
            summary += f"  • 主要原料：{product.main_raw_materials or '-'}\n"
            summary += f"  • 規格：{product.product_standard or '-'}\n"
            summary += f"  • 技術優勢：{product.technical_advantages or '-'}\n"

        return summary

    def get_initial_greeting(self) -> str:
        """Get the initial greeting with menu options"""
        # Check if user has existing data
        existing_data = self.db.query(CompanyOnboarding).filter(
            CompanyOnboarding.user_id == self.user_id,
            CompanyOnboarding.is_current == True
        ).first()

        if existing_data and existing_data.industry:
            # Calculate progress (6 fields total, ESG counts as one)
            fields_done = 0
            total_fields = 6
            if existing_data.industry:
                fields_done += 1
            if existing_data.capital_amount is not None:
                fields_done += 1
            if existing_data.invention_patent_count is not None:
                fields_done += 1
            if existing_data.utility_patent_count is not None:
                fields_done += 1
            if existing_data.certification_count is not None:
                fields_done += 1
            # ESG counts as ONE field
            if existing_data.esg_certification:
                fields_done += 1

            # Build missing fields list
            missing_fields = []
            if not existing_data.industry:
                missing_fields.append("產業別")
            if existing_data.capital_amount is None:
                missing_fields.append("資本總額")
            if existing_data.invention_patent_count is None:
                missing_fields.append("發明專利數量")
            if existing_data.utility_patent_count is None:
                missing_fields.append("新型專利數量")
            if existing_data.certification_count is None:
                missing_fields.append("公司認證資料")
            if not existing_data.esg_certification:
                missing_fields.append("ESG相關認證")

            missing_str = ""
            if missing_fields:
                missing_str = f"\n\n⚠️ 尚未填寫的資料：{', '.join(missing_fields)}"

            products_count = len(existing_data.products) if existing_data.products else 0

            # User has existing data
            return f"""您好！歡迎回來！🤖

══════════════════════════════
📊 資料填寫進度：【{fields_done}/{total_fields} 已完成】
══════════════════════════════
• 產業別：{existing_data.industry or '未填寫'}
• 資本額：{existing_data.capital_amount or '未填寫'} 臺幣
• 發明專利：{existing_data.invention_patent_count if existing_data.invention_patent_count is not None else '未填寫'} 件
• 新型專利：{existing_data.utility_patent_count if existing_data.utility_patent_count is not None else '未填寫'} 件
• 公司認證：{existing_data.certification_count if existing_data.certification_count is not None else '未填寫'} 項
• ESG認證：{existing_data.esg_certification or '未填寫'}
• 產品數量：{products_count} 項{missing_str}

💡 完整資料可解鎖平臺功能：
   • 【推薦引擎】- 曝光產品、尋找合作夥伴
   • 【補助引擎】- 協助申請政府補助案

══════════════════════════════
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

══════════════════════════════
📋 為什麼需要填寫公司資料？
══════════════════════════════
填寫完整的公司資料可以幫助我們：
✅ 了解貴公司的產業屬性與優勢
✅ 透過【推薦引擎】幫助您曝光產品、尋找合作夥伴
✅ 使用【補助引擎】協助申請政府補助案
✅ 精準配對商業機會與資源

══════════════════════════════
📝 我們需要收集的資料：
══════════════════════════════
【基本資料】共6項：
1️⃣ 產業別
2️⃣ 資本總額
3️⃣ 發明專利數量
4️⃣ 新型專利數量
5️⃣ 公司認證資料
6️⃣ ESG相關認證

【產品資訊】填完基本資料後收集

💡 您可以用自然的方式回答，也可以上傳文件讓系統自動提取資料。

══════════════════════════════
讓我們開始吧！【進度：0/6 已完成】
請問貴公司所屬的產業別是什麼？
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

    def get_state_aware_extraction_prompt(self) -> str:
        """
        Get a focused extraction prompt based on current stage.
        This is the KEY fix: tell AI exactly what ONE field to extract.
        """
        stage = self.get_current_stage()
        progress = self.get_progress()
        fields_done = progress['fields_completed']

        # Get existing products for context
        existing_products = []
        if self.onboarding_data and self.onboarding_data.products:
            existing_products = [p.product_id for p in self.onboarding_data.products if p.product_id]

        products_context = ""
        if existing_products:
            products_context = f"\n現有產品ID列表：{', '.join(existing_products)}"

        if stage == OnboardingStage.PRODUCT:
            # Product collection mode
            product_field = self.onboarding_data.current_product_field or ProductField.PRODUCT_ID
            field_name = self.PRODUCT_FIELD_TO_DISPLAY_NAME.get(product_field, "產品資訊")
            draft = self.get_product_draft()
            field_index = self.PRODUCT_FIELD_ORDER.index(product_field) + 1 if product_field in self.PRODUCT_FIELD_ORDER else 1

            draft_summary = ""
            if draft:
                draft_summary = "\n目前產品草稿：\n"
                for k, v in draft.items():
                    display_k = self.PRODUCT_FIELD_TO_DISPLAY_NAME.get(ProductField(k), k)
                    draft_summary += f"  • {display_k}: {v}\n"

            return f"""你是一個資料提取助理。

🎯 目前正在收集的欄位：**{field_name}**
📊 產品進度：【{field_index}/6 已填寫】
{draft_summary}{products_context}

📌 可用的工具：
1. **collect_product_field** - 當使用者只提供單一欄位時使用
2. **add_complete_product** - 當使用者一次提供完整產品資訊（6個欄位全部）時使用
3. **update_product** - 當使用者說要「修改」、「更新」、「更改」某個產品時使用
4. **update_company_field** - 當使用者說要「修改」、「更新」公司基本資料（如資本額、專利數量等）時使用
5. **view_data** - 當使用者說「列出」、「顯示」、「查看」資料時使用
6. **mark_completed** - 當使用者說「完成」、「結束」、「不用了」時使用

⚠️ 重要判斷規則：
- 如果使用者說「列出」、「顯示」、「查看」公司資料 → 使用 view_data(data_type="company")
- 如果使用者說「列出」、「顯示」、「查看」產品資料 → 使用 view_data(data_type="products")
- 如果使用者說「列出」、「顯示」、「查看」全部資料 → 使用 view_data(data_type="all")
- 如果使用者說要修改「公司資料」、「資本額」、「專利」等基本資料 → 使用 update_company_field
  - field 可選：industry, capital_amount, invention_patent_count, utility_patent_count, certification_count, esg_certification
  - 例如「資本額改成300萬」→ update_company_field(field="capital_amount", value="3000000")
- 如果使用者說「修改」、「更新」、「更改」某產品的某欄位 → 使用 update_product
- 如果使用者訊息包含「產品ID」+「產品名稱」+「價格」+「主要原料」+「規格」+「技術優勢」→ 使用 add_complete_product
- 如果使用者只提供單一值（回答當前問題）→ 使用 collect_product_field，field="{product_field.value}"
- 如果使用者回答「-」、「無」、「沒有」→ 使用 collect_product_field，value="-"

回覆時請友善確認已記錄的資訊。"""

        elif stage == OnboardingStage.COMPLETED:
            return f"""你是一個資料收集助理。使用者已完成基本資料收集。
{products_context}

📌 可用的工具：
1. **update_company_field** - 當使用者說要「修改」、「更新」公司基本資料時使用
   - 可更新欄位：industry, capital_amount, invention_patent_count, utility_patent_count, certification_count, esg_certification
2. **update_product** - 當使用者說要「修改」、「更新」某個產品時使用
   - 需要指定 product_id 和要更新的 field
3. **add_complete_product** - 當使用者要新增產品時使用
4. **view_data** - 當使用者說「列出」、「顯示」、「查看」資料時使用
5. **mark_completed** - 當使用者確認完成時使用

⚠️ 重要判斷規則：
- 如果使用者說「列出」、「顯示」、「查看」公司資料 → 使用 view_data(data_type="company")
- 如果使用者說「列出」、「顯示」、「查看」產品資料 → 使用 view_data(data_type="products")
- 如果使用者說「列出」、「顯示」、「查看」全部資料 → 使用 view_data(data_type="all")
- 如果使用者說「修改產品X的價格為Y」→ 使用 update_product(product_id="X", field="price", value="Y")
- 如果使用者說「修改資本額為Y」→ 使用 update_company_field(field="capital_amount", value="Y")
- 如果使用者說「新增產品」並提供完整資訊 → 使用 add_complete_product
- 如果使用者說「完成」、「結束」→ 使用 mark_completed

回覆時請友善確認已更新的資訊，並顯示更新後的結果。"""

        else:
            # Company data collection mode
            field_name = self.STAGE_TO_DISPLAY_NAME.get(stage, "資料")
            field_key = self.STAGE_TO_FIELD.get(stage, "")

            # Special handling for different fields
            field_hints = {
                OnboardingStage.INDUSTRY: "例如：食品業、鋼鐵業、電子業、資訊服務業等",
                OnboardingStage.CAPITAL_AMOUNT: "請轉換為臺幣數字，例如「500萬」→ 5000000",
                OnboardingStage.INVENTION_PATENT_COUNT: "請提取數量，例如「5個」→ 5",
                OnboardingStage.UTILITY_PATENT_COUNT: "請提取數量，例如「3個」→ 3",
                OnboardingStage.CERTIFICATION_COUNT: "不包括ESG認證，例如 ISO 9001, HACCP 等的數量",
                OnboardingStage.ESG_CERTIFICATION: "如 ISO 14064, ISO 14067 等，需同時提供認證列表和數量",
            }

            hint = field_hints.get(stage, "")

            return f"""你是一個資料提取助理。

🎯 目前正在收集的欄位：**{field_name}**
📊 基本資料進度：【{fields_done}/6 已完成】

⚠️ 重要規則：
1. 你必須調用 update_company_data 函數
2. 只提取 {field_key} 這一個欄位
3. 不要提取或猜測其他欄位
4. {hint}
5. 如果使用者回答「無」、「沒有」、「0」，設置對應的值（字串設為「無」，數字設為 0）

{"⚠️ ESG認證特別注意：必須同時傳遞 esg_certification（認證列表字串）和 esg_certification_count（認證數量）" if stage == OnboardingStage.ESG_CERTIFICATION else ""}

回覆時請友善確認已記錄的資訊，並顯示進度。"""

    def get_state_aware_tools(self) -> list:
        """Get tool definitions based on current stage"""
        stage = self.get_current_stage()

        # Common tool for updating existing products (available in PRODUCT and COMPLETED stages)
        update_product_tool = {
            "type": "function",
            "function": {
                "name": "update_product",
                "description": "更新現有產品的資訊（當使用者說要修改、更新某個產品時使用）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "description": "要更新的產品ID"},
                        "field": {
                            "type": "string",
                            "description": "要更新的欄位",
                            "enum": ["product_name", "price", "main_raw_materials", "product_standard", "technical_advantages"]
                        },
                        "value": {"type": "string", "description": "新的值"}
                    },
                    "required": ["product_id", "field", "value"]
                }
            }
        }

        # Tool for adding a complete product at once (when user provides all info)
        add_complete_product_tool = {
            "type": "function",
            "function": {
                "name": "add_complete_product",
                "description": "當使用者一次提供完整產品資訊時使用（包含所有6個欄位）",
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
                    "required": ["product_id", "product_name", "price", "main_raw_materials", "product_standard", "technical_advantages"]
                }
            }
        }

        # Tool for updating company fields (available in PRODUCT and COMPLETED stages)
        update_company_field_tool = {
            "type": "function",
            "function": {
                "name": "update_company_field",
                "description": "更新公司基本資料的某個欄位（當使用者說要修改公司資本額、專利數量等基本資料時使用）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "field": {
                            "type": "string",
                            "description": "要更新的欄位",
                            "enum": ["industry", "capital_amount", "invention_patent_count", "utility_patent_count", "certification_count", "esg_certification"]
                        },
                        "value": {"type": "string", "description": "新的值"}
                    },
                    "required": ["field", "value"]
                }
            }
        }

        # Tool for viewing data (available in PRODUCT and COMPLETED stages)
        view_data_tool = {
            "type": "function",
            "function": {
                "name": "view_data",
                "description": "當使用者要求「列出」、「顯示」、「查看」公司資料或產品資料時使用",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_type": {
                            "type": "string",
                            "description": "要查看的資料類型",
                            "enum": ["company", "products", "all"]
                        }
                    },
                    "required": ["data_type"]
                }
            }
        }

        if stage == OnboardingStage.PRODUCT:
            # Product field collection - single field at a time
            product_field = self.onboarding_data.current_product_field or ProductField.PRODUCT_ID
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "collect_product_field",
                        "description": f"收集產品的 {self.PRODUCT_FIELD_TO_DISPLAY_NAME.get(product_field, '資訊')}（當使用者只提供單一欄位時使用）",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "field": {
                                    "type": "string",
                                    "description": "欄位名稱",
                                    "enum": [product_field.value]
                                },
                                "value": {
                                    "type": "string",
                                    "description": f"使用者提供的{self.PRODUCT_FIELD_TO_DISPLAY_NAME.get(product_field, '值')}"
                                }
                            },
                            "required": ["field", "value"]
                        }
                    }
                },
                add_complete_product_tool,  # Allow bulk product input
                update_product_tool,  # Allow updating existing products
                update_company_field_tool,  # Allow updating company data
                view_data_tool,  # Allow viewing data
                {
                    "type": "function",
                    "function": {
                        "name": "mark_completed",
                        "description": "僅當使用者明確說「完成」、「結束」、「不用了」時調用",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "completed": {"type": "boolean"}
                            },
                            "required": ["completed"]
                        }
                    }
                }
            ]
        elif stage == OnboardingStage.COMPLETED:
            # Allow updates and viewing in completed stage
            return [
                update_company_field_tool,  # Allow updating company data
                update_product_tool,  # Allow updating existing products
                add_complete_product_tool,  # Allow adding new products
                view_data_tool,  # Allow viewing data
                {
                    "type": "function",
                    "function": {
                        "name": "mark_completed",
                        "description": "確認完成資料收集",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "completed": {"type": "boolean"}
                            },
                            "required": ["completed"]
                        }
                    }
                }
            ]
        else:
            # Company data collection - stage-specific tool
            field_key = self.STAGE_TO_FIELD.get(stage, "industry")
            field_name = self.STAGE_TO_DISPLAY_NAME.get(stage, "資料")

            # Build properties based on current stage
            properties = {}
            required = []

            if stage == OnboardingStage.INDUSTRY:
                properties["industry"] = {"type": "string", "description": "產業別"}
                required = ["industry"]
            elif stage == OnboardingStage.CAPITAL_AMOUNT:
                properties["capital_amount"] = {"type": "integer", "description": "資本總額（臺幣）"}
                required = ["capital_amount"]
            elif stage == OnboardingStage.INVENTION_PATENT_COUNT:
                properties["invention_patent_count"] = {"type": "integer", "description": "發明專利數量"}
                required = ["invention_patent_count"]
            elif stage == OnboardingStage.UTILITY_PATENT_COUNT:
                properties["utility_patent_count"] = {"type": "integer", "description": "新型專利數量"}
                required = ["utility_patent_count"]
            elif stage == OnboardingStage.CERTIFICATION_COUNT:
                properties["certification_count"] = {"type": "integer", "description": "公司認證數量（不含ESG）"}
                required = ["certification_count"]
            elif stage == OnboardingStage.ESG_CERTIFICATION:
                properties["esg_certification"] = {"type": "string", "description": "ESG認證列表"}
                properties["esg_certification_count"] = {"type": "integer", "description": "ESG認證數量"}
                required = ["esg_certification", "esg_certification_count"]

            return [
                {
                    "type": "function",
                    "function": {
                        "name": "update_company_data",
                        "description": f"更新 {field_name}",
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required
                        }
                    }
                }
            ]

    def extract_data_with_ai(self, user_message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """
        Use OpenAI to extract structured data from conversation.
        KEY CHANGE: Uses state-aware prompt and tools to extract only ONE field at a time.
        """
        client = get_openai_client()
        if not client:
            return {"error": "OpenAI API key not configured"}

        # Get state-aware prompt and tools
        extraction_prompt = self.get_state_aware_extraction_prompt()
        tools = self.get_state_aware_tools()

        # Build conversation for OpenAI
        messages = [
            {"role": "system", "content": extraction_prompt},
            {"role": "system", "content": f"目前已收集的資料：\n{self.get_current_data_summary()}"}
        ]

        # Add recent conversation history (last 5 messages for context)
        for msg in conversation_history[-5:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        try:
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                tools=tools,
                tool_choice="required"  # FORCE tool call - don't allow text-only response
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

    def _count_esg_certifications(self, certification_str: str) -> int:
        """
        Count the number of ESG certifications from a comma-separated string.
        Handles common separators and trims whitespace.
        """
        if not certification_str or certification_str.strip().lower() in ["無", "没有", "none", "-"]:
            return 0

        # Split by common separators: comma, Chinese comma, semicolon, newline
        import re
        certifications = re.split(r'[,，;；\n]+', certification_str)
        # Filter out empty strings and trim whitespace
        certifications = [c.strip() for c in certifications if c.strip()]
        return len(certifications)

    def update_onboarding_data(self, data: Dict[str, Any]) -> bool:
        """
        Update onboarding data with extracted information.

        IMPORTANT: ESG is atomically updated - when esg_certification is provided,
        the count is automatically calculated. Never allow partial ESG writes.
        """
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

            # ============ ATOMIC ESG UPDATE ============
            # When esg_certification is provided, automatically calculate and set the count.
            # This ensures both fields are always in sync - never allow partial ESG writes.
            if "esg_certification" in data and data["esg_certification"]:
                esg_str = str(data["esg_certification"])
                self.onboarding_data.esg_certification = esg_str
                # Automatically calculate count from the certification string
                self.onboarding_data.esg_certification_count = self._count_esg_certifications(esg_str)
                updated = True
            # Note: We intentionally ignore esg_certification_count if passed separately
            # to enforce atomic updates. The count is ALWAYS derived from the string.

            if updated:
                self.db.commit()

            return updated

        except Exception as e:
            print(f"Error updating onboarding data: {e}")
            self.db.rollback()
            return False

    def add_product(self, product_data: Dict[str, Any]) -> tuple[Optional[Product], bool, List[str]]:
        """
        Add or update a product in the onboarding data with duplicate checking
        Returns: (product, was_updated, missing_fields)
        - product: The created/updated product, or None if validation failed
        - was_updated: True if existing product was updated
        - missing_fields: List of required fields that are missing
        """
        try:
            # Validate ALL required fields - all 6 fields must be provided
            required_fields = {
                "product_id": "產品ID",
                "product_name": "產品名稱",
                "price": "價格",
                "main_raw_materials": "主要原料",
                "product_standard": "產品規格",
                "technical_advantages": "技術優勢"
            }
            missing_fields = []
            for field, display_name in required_fields.items():
                if not product_data.get(field):
                    missing_fields.append(display_name)

            # If any required fields are missing, don't create the product
            if missing_fields:
                return None, False, missing_fields

            # Check for duplicate product_id in current onboarding
            product_id = product_data.get("product_id")
            if product_id:
                existing_product = self.db.query(Product).filter(
                    Product.onboarding_id == self.onboarding_data.id,
                    Product.product_id == product_id
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
                    return existing_product, True, []  # Return True indicating update

            # Create new product
            product = Product(
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
            return product, False, []  # Return False indicating new product
        except Exception as e:
            print(f"Error adding product: {e}")
            self.db.rollback()
            return None, False, []

    def get_next_field_question(self) -> str:
        """Get the next field question based on what's already collected"""
        # Refresh data from database to get the latest state
        self.db.refresh(self.onboarding_data)

        # Calculate progress
        progress = self.get_progress()
        fields_done = progress['fields_completed']
        total_fields = progress['total_fields']
        remaining = total_fields - fields_done
        progress_str = f"【進度：{fields_done}/{total_fields} 已完成】"

        # Check fields in order and return the first missing one
        if not self.onboarding_data.industry:
            return f"{progress_str}\n請問您的公司所屬產業別是什麼？（例如：食品業、鋼鐵業、電子業等）"

        if self.onboarding_data.capital_amount is None:
            return f"{progress_str}\n請問您的公司資本總額是多少？（以臺幣為單位）"

        if self.onboarding_data.invention_patent_count is None:
            return f"{progress_str}\n請問貴公司有多少**發明專利**？（請提供數量）\n\n💡 發明專利是什麼？\n發明專利是針對「技術方案」的專利，包括產品發明（如新材料、新裝置）或方法發明（如製程、配方）。保護期限為20年，是技術創新能力的重要指標。"

        if self.onboarding_data.utility_patent_count is None:
            return f"{progress_str}\n請問貴公司有多少**新型專利**？（請提供數量）\n\n💡 新型專利是什麼？\n新型專利是針對產品「形狀、構造」的專利，例如機械結構改良、零件設計等。保護期限為10年，審查較快速，適合產品外觀或結構的創新。"

        if self.onboarding_data.certification_count is None:
            return f"{progress_str}\n請問貴公司有多少公司認證資料？（不包括ESG認證，例如：ISO 9001、HACCP等）"

        if not self.onboarding_data.esg_certification:
            return f"{progress_str}\n請列出貴公司所有ESG相關認證（例如：ISO 14064, ISO 14067, ISO 14046）。如果沒有，請回答「無」。"

        # All basic fields collected, ask for products
        products_count = self.db.query(Product).filter(
            Product.onboarding_id == self.onboarding_data.id
        ).count()

        if products_count == 0:
            # Build complete basic data summary
            basic_data_summary = f"""🎉 太好了！基本資料已收集完成 {progress_str}

══════════════════════════════
📋 基本資料摘要
══════════════════════════════
• 產業別：{self.onboarding_data.industry or '未填寫'}
• 資本額：{self.onboarding_data.capital_amount or '未填寫'} 臺幣
• 發明專利：{self.onboarding_data.invention_patent_count if self.onboarding_data.invention_patent_count is not None else '未填寫'} 件
• 新型專利：{self.onboarding_data.utility_patent_count if self.onboarding_data.utility_patent_count is not None else '未填寫'} 件
• 公司認證：{self.onboarding_data.certification_count if self.onboarding_data.certification_count is not None else '未填寫'} 項
• ESG認證：{self.onboarding_data.esg_certification or '未填寫'}

接下來請提供產品資訊，讓【推薦引擎】能幫助您曝光產品。

我會逐一詢問每個產品的詳細資訊（共6項）：
• 產品ID → 產品名稱 → 價格 → 主要原料 → 規格 → 技術優勢
（如果有多個產品，建議直接跟著格式上傳檔案）

請先提供第一個產品的**產品ID**（例如：PROD001）
【產品進度：0/6 已填寫】"""
            return basic_data_summary
        else:
            # Include product summary
            products_summary = self.get_products_summary()
            return f"📦 目前已新增 {products_count} 個產品。{progress_str}{products_summary}\n\n還有其他產品要新增嗎？如果要新增，請提供新產品的**產品ID** 開始流程或直接上傳文件 （PDF、Word）即可。\n如果資料已完成，請告訴我「完成」。\n\n💡 產品資訊越完整，【推薦引擎】越能精準幫您配對商機！"

    def process_message(self, user_message: str) -> tuple[str, bool]:
        """
        Process user message with AI and return bot response.
        Uses state machine for deterministic field collection.
        Returns: (response_message, is_completed)
        """
        # Sync stage with actual data state (in case of data import or state mismatch)
        self.sync_stage_with_data()

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

        # Get current stage info for validation
        current_stage = self.get_current_stage()
        expected_field = self.get_expected_field()
        expected_tool = self.get_expected_tool()

        # Extract data with AI (state-aware extraction)
        ai_result = self.extract_data_with_ai(user_message, conversation_history)

        if "error" in ai_result:
            # If extraction failed, ask for the field again
            stage_name = self.STAGE_TO_DISPLAY_NAME.get(current_stage, "資料")
            return f"抱歉，我無法理解您的回答。請再次提供 **{stage_name}**。", False

        # Process function calls with state machine
        completed = False
        data_updated = False
        product_field_collected = False
        product_just_saved = False  # Track when a product was just saved
        view_data_requested = False  # Track when view_data was requested
        view_data_response = ""  # Store view_data response

        function_calls = ai_result.get("function_calls", [])

        # Validate that we got a tool call
        if not function_calls:
            # No tool call - ask for the expected field again
            stage_name = self.STAGE_TO_DISPLAY_NAME.get(current_stage, "資料")
            if current_stage == OnboardingStage.PRODUCT:
                product_field = self.onboarding_data.current_product_field or ProductField.PRODUCT_ID
                field_name = self.PRODUCT_FIELD_TO_DISPLAY_NAME.get(product_field, "產品資訊")
                return f"請提供 **{field_name}**", False
            else:
                return f"請提供 **{stage_name}**", False

        for call in function_calls:
            tool_name = call["name"]
            args = call["arguments"]

            if tool_name == "update_company_data":
                # Update the specific field and advance stage
                if self.update_onboarding_data(args):
                    data_updated = True
                    # Advance to next stage
                    self.advance_stage()

            elif tool_name == "collect_product_field":
                # Collect single product field
                field = args.get("field")
                value = args.get("value")

                if field and value:
                    # Update product draft
                    self.update_product_draft(field, value)
                    product_field_collected = True

                    # Check if product is complete
                    if self.is_product_draft_complete():
                        # Save product from draft (this resets the draft!)
                        product = self.save_product_from_draft()
                        if product:
                            # Mark that product was just saved for response generation
                            product_just_saved = True
                    else:
                        # Advance to next product field
                        self.advance_product_field()

            elif tool_name == "mark_completed":
                if args.get("completed"):
                    self.session.status = ChatSessionStatus.COMPLETED
                    self.onboarding_data.current_stage = OnboardingStage.COMPLETED
                    self.db.commit()
                    completed = True

            elif tool_name == "add_complete_product":
                # Add a complete product at once (bulk input)
                product_data = {
                    "product_id": args.get("product_id"),
                    "product_name": args.get("product_name"),
                    "price": args.get("price"),
                    "main_raw_materials": args.get("main_raw_materials"),
                    "product_standard": args.get("product_standard"),
                    "technical_advantages": args.get("technical_advantages")
                }
                product, was_updated, missing = self.add_product(product_data)
                if product:
                    product_just_saved = True
                    # Reset draft state for next product
                    self.reset_product_draft()

            elif tool_name == "update_product":
                # Update an existing product
                product_id = args.get("product_id")
                field = args.get("field")
                value = args.get("value")

                if product_id and field and value:
                    # Find the product
                    product = self.db.query(Product).filter(
                        Product.onboarding_id == self.onboarding_data.id,
                        Product.product_id == product_id
                    ).first()

                    if product:
                        # Update the field
                        if field == "product_name":
                            product.product_name = value
                        elif field == "price":
                            product.price = value
                        elif field == "main_raw_materials":
                            product.main_raw_materials = value
                        elif field == "product_standard":
                            product.product_standard = value
                        elif field == "technical_advantages":
                            product.technical_advantages = value
                        self.db.commit()
                        data_updated = True

            elif tool_name == "update_company_field":
                # Update a company field (in COMPLETED stage)
                field = args.get("field")
                value = args.get("value")

                if field and value:
                    update_data = {}
                    if field == "industry":
                        update_data["industry"] = value
                    elif field == "capital_amount":
                        update_data["capital_amount"] = int(value) if value.isdigit() else int(float(value))
                    elif field == "invention_patent_count":
                        update_data["invention_patent_count"] = int(value)
                    elif field == "utility_patent_count":
                        update_data["utility_patent_count"] = int(value)
                    elif field == "certification_count":
                        update_data["certification_count"] = int(value)
                    elif field == "esg_certification":
                        update_data["esg_certification"] = value

                    if update_data:
                        self.update_onboarding_data(update_data)
                        data_updated = True

            elif tool_name == "view_data":
                # View company or product data
                data_type = args.get("data_type", "all")
                view_data_requested = True

                if data_type == "company":
                    view_data_response = self.get_company_summary()
                elif data_type == "products":
                    view_data_response = self.get_products_summary() or "目前尚未新增任何產品。"
                else:  # all
                    view_data_response = f"{self.get_company_summary()}\n\n{self.get_products_summary()}"

                # Add prompt for what to do next
                view_data_response += "\n\n還需要修改資料嗎？或說「完成」結束。"

        # Generate response based on new state
        response_message = ai_result.get("message", "")

        # Handle view_data request
        if view_data_requested:
            return view_data_response, False

        if completed:
            return response_message or "感謝您完成資料收集！您的公司資料已成功儲存。", True

        if not response_message:
            # Generate confirmation and next question
            progress = self.get_progress()
            fields_done = progress['fields_completed']
            total_fields = progress['total_fields']
            new_stage = self.get_current_stage()

            if product_just_saved:
                # Product was just saved (either from bulk input or single field collection)
                products_count = len(self.onboarding_data.products) if self.onboarding_data.products else 0
                response_message = f"✅ 產品已成功新增！\n\n{self.get_products_summary()}\n\n還有其他產品要新增嗎？請提供新產品的**產品ID**，或說「完成」結束。"

            elif data_updated:
                # Data was updated
                if current_stage == OnboardingStage.COMPLETED:
                    # Update in COMPLETED stage - show summary and ask what else to do
                    response_message = f"✅ 已更新資料！\n\n{self.get_current_data_summary()}\n\n{self.get_products_summary()}\n\n還需要修改其他資料嗎？或說「完成」結束。"
                else:
                    # Company field was collected during onboarding
                    stage_name = self.STAGE_TO_DISPLAY_NAME.get(current_stage, "資料")
                    if fields_done == total_fields:
                        # All basic fields complete, transition to product
                        confirmation = f"✅ 已記錄 {stage_name}！\n\n"
                    elif fields_done >= total_fields - 2:
                        confirmation = f"✅ 已記錄 {stage_name}！【進度：{fields_done}/{total_fields} 已完成】再 {total_fields - fields_done} 項就完成基本資料了！\n\n"
                    else:
                        confirmation = f"✅ 已記錄 {stage_name}！【進度：{fields_done}/{total_fields} 已完成，還剩 {total_fields - fields_done} 項】\n\n"

                    next_question = self.get_next_field_question()
                    response_message = confirmation + next_question

            elif product_field_collected:
                # Product field was collected (single field mode)
                product_field = self.onboarding_data.current_product_field or ProductField.PRODUCT_ID
                draft = self.get_product_draft()
                filled_count = len([v for v in draft.values() if v])
                field_name = self.PRODUCT_FIELD_TO_DISPLAY_NAME.get(product_field, "資訊")
                response_message = f"✅ 已記錄！【產品進度：{filled_count}/6 已填寫】\n\n請提供 **{field_name}**"
            else:
                # Fallback
                response_message = self.get_next_field_question()

        return response_message, completed

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress of data collection"""
        fields_completed = 0
        total_fields = 6  # Total number of company fields: industry, capital, 2 patents, certification, esg (as one)

        # Only collect fields within chatbot's responsibility
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
        # ESG counts as ONE field (either esg_certification_count or esg_certification being filled)
        if self.onboarding_data.esg_certification:
            fields_completed += 1

        return {
            "company_info_complete": fields_completed == total_fields,
            "fields_completed": fields_completed,
            "total_fields": total_fields,
            "products_count": len(self.onboarding_data.products) if self.onboarding_data.products else 0
        }
