import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 確保載入環境變數
load_dotenv()

class InsuranceRagEngine:
    def __init__(self, persist_dir: str = None):
        """
        初始化 RAG 引擎
        :param persist_dir: 指定資料庫路徑。若為 None，則嘗試從 .env讀取，預設為 ./chroma_db
        """
        self.api_key = os.getenv("gemini_key")
        if not self.api_key:
            raise ValueError("❌ 錯誤: 環境變數 'gemini_key' 未設定")

        # 1. 決定資料庫路徑 (修復路徑錯亂問題的核心)
        if persist_dir is None:
            self.persist_dir = os.getenv("persist_dir", "./chroma_db")
        else:
            self.persist_dir = persist_dir

        print(f"🔧 [Init] 正在載入向量資料庫，路徑: {os.path.abspath(self.persist_dir)}")

        # 2. 初始化 Embedding (必須與建庫時使用的模型一致)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", # 確保使用 004
            google_api_key=self.api_key
        )
        
        # 3. 初始化向量資料庫
        # 注意: 這裡不會檢查路徑是否存在，若路徑錯了會建立一個空的 DB
        self.vectorstore = Chroma(
            collection_name="Travel_Insurance_RAG",
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )

        # 檢查資料庫是否真的有資料 (自我診斷)
        count = self.vectorstore._collection.count()
        if count == 0:
            print(f"⚠️ 警告: 載入的資料庫 '{self.persist_dir}' 是空的！請檢查路徑或重新建庫。")
        else:
            print(f"✅ 資料庫載入成功，目前共有 {count} 筆資料。")
        
        # 4. 初始化 LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0
        )
        
        self.setup_prompts()

    def setup_prompts(self):
        # 1. 查詢生成 Prompt
        self.query_gen_template = """
        你是一個精通保險條款的專業分析師。使用者的問題可能包含口語或模糊概念。
        請根據使用者的問題，產生 4 個具備「高語意差異」的搜尋查詢，以確保檢索能覆蓋所有潛在條款。

        產生規則：
        1. **術語精確化**：將口語轉換為標準術語（如「飛機遲到」轉為「班機延誤」、「旅程延誤」）。
        2. **概念辨析 (核心)**：若問題涉及「延誤」或「損失」，請分別針對「旅程/班機」與「行李」產生獨立查詢，避免兩者混淆。
           - 範例：若問延誤，應同時搜尋「旅程延誤承保範圍」與「班機延誤給付」。
        3. **維度擴展**：查詢應涵蓋「承保事故」、「理賠標準」、「理賠文件」與「除外責任」四個維度。
        4. **排除干擾**：若使用者未提及「行李」，查詢應側重於「旅程」與「交通工具」本身。

        使用者問題：{question}

        請直接列出 4 個查詢，每行一個，不要有編號或額外文字：
        """
        self.query_gen_prompt = PromptTemplate(
            template=self.query_gen_template,
            input_variables=["question"]
        )

        # 2. 回答生成 Prompt
        self.qa_template = """
        你是一位資深的保險理賠顧問。請根據保險條款內容回答使用者的理賠問題。
        
        【條款內容】:
        {context}
        
        使用者問題：{question}
        
        回答規範（請嚴格遵守）：
        1. **自然對話**：請直接針對問題給出建議，禁止使用「根據您提供的 Context」、「根據參考資料」、「根據檢索到的內容」或「在本次提供的文件中」等類似術語。
        2. **專業立場**：你的回答應像是在面對面諮詢客戶，語氣要專業且誠懇，直接引用條款名稱與條號（例如：根據第 XX 條規定...）。
        3. **完整性**：請綜合資訊，不要遺漏重要條款，特別是「特別不保事項」或「除外責任」。
        4. **誠實處理**：如果條款中確實沒有提到相關資訊，請委婉告知目前條款內容未涵蓋此項目，不要編造答案。
        
        請直接開始回答：
        """
        self.qa_prompt = PromptTemplate(
            template=self.qa_template,
            input_variables=["context", "question"]
        )

    def generate_search_queries(self, question: str) -> list[str]:
        """利用 LLM 產生多樣化的搜尋字串"""
        chain = self.query_gen_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"question": question})
        queries = [q.strip() for q in result.split('\n') if q.strip()]
        return queries[:5]

    def get_answer(self, user_question: str, chat_history_list: list = []):
        # --- 步驟 1: 生成多重查詢 ---
        generated_queries = self.generate_search_queries(user_question)
        if user_question not in generated_queries:
            generated_queries.insert(0, user_question)

        print(f"\n[DEBUG] 原始問題: {user_question}")
        print(f"\n[DEBUG] 執行搜尋策略: {generated_queries}")
        # --- 步驟 2: 執行多重檢索 (MMR) ---
        
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6, 
                "fetch_k": 30, 
                "lambda_mult": 0.5
            }
        )
        
        # 使用字典來儲存，避免簡單 extend 導致的順序偏差
        query_results = []
        for query in generated_queries:
            docs = retriever.invoke(query)
            query_results.append(docs)

        # --- 步驟 3: 智慧交錯排序 (Interleaving) ---
        # 確保每個搜尋關鍵字抓到的第一名都有機會進入最終 Context
        all_docs = []
        max_docs_per_query = 8
        for i in range(max_docs_per_query):
            for docs in query_results:
                if i < len(docs):
                    all_docs.append(docs[i])

        # --- 步驟 4: 去除重複文件 ---
        unique_docs = []
        seen_content = set()
        
        for doc in all_docs:
            content_snippet = doc.page_content.strip()
            # 使用全文雜湊或條號作為去重標準
            doc_id = doc.metadata.get('section_id', doc.metadata.get('article_no', content_snippet[:100]))
            
            if content_snippet not in seen_content:
                unique_docs.append(doc)
                seen_content.add(content_snippet)
        
        # 擴大 Context 視窗 (Gemini 2.5 Flash 處理能力強，可以多給一點資料)
        final_docs = unique_docs[:18] 
        
        print(f"[DEBUG] 最終參與回答的條款片段數: {len(final_docs)}")
        # 除錯：印出目前抓到的條號，確認「班機延誤」有沒有進來
        retrieved_articles = [d.metadata.get('article_no', 'Unknown') for d in final_docs]
        print(f"[DEBUG] 檢索到的條號清單: {retrieved_articles}")

        # --- 步驟 5: 組裝 Context ---
        def format_doc(doc):
            meta = doc.metadata
            source = meta.get('source', '保險條款')
            sec_id = meta.get('section_id', meta.get('article_no', '無條號'))
            title = meta.get('title', meta.get('article_title', '無標題'))
            return f"📄 來源：{source} | ⚖️ 條號：{sec_id} | 📝 標題：{title}\n📖 內容：{doc.page_content}"

        context_text = "\n--------------------\n".join([format_doc(d) for d in final_docs])

        if not context_text:
            return {"answer": "目前的條款資料中未查獲相關內容...", "source_documents": [], "debug_queries": generated_queries}

        # --- 步驟 6: 生成回答 ---
        chain = self.qa_prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context_text, "question": user_question})
        
        return {"answer": response, "source_documents": final_docs, "debug_queries": generated_queries}