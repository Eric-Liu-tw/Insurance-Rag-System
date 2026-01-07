# 旅遊不便險 AI 語意檢索助手 (Insurance RAG System)

本專案利用 **LangChain** 框架結合 **Google Gemini 2.5 Flash** 模型，構建了一個針對「旅遊不便險」條款的檢索增強生成（RAG）系統。使用者可以透過自然語言詢問保險理賠細節，系統會自動從保險契約文件中檢索相關條文並給出精準回答。

## 🚀 核心功能
* **自然語言問答**：透過 LLM 理解使用者問題，解決保險條款生澀難懂的痛點。
* **精準條文檢索**：自動從 PDF/Docx 文件中提取資訊，並標註參考來源與條號。
* **對話歷史保存**：具備 Session 管理功能，支援多輪對話上下文理解。
* **視覺化 Dashboard**：使用 Streamlit 打造直觀的 Web UI 互動界面 。

## 🛠️ 技術棧
* **語言/框架**：Python, LangChain 
* **AI 模型**：Google Gemini 2.5 Flash (Generative AI)
* **向量資料庫**：ChromaDB (Vector Store)
* **資料處理**：Pandas, Regex, RecursiveCharacterTextSplitter
* **部署容器**：Docker (具備容器化部署基礎) 

## 🏗️ 系統架構與實作細節

### 1. 資料預處理 (ETL 流程)
* 使用正則表達式 (Regex) 進行資料清洗，針對保險條文格式進行正規化處理，確保「第 X 條」等標題能被正確識別。
* 實作 Metadata 標記策略，將文件來源與條號注入向量塊中，提升檢索的可信度。

### 2. 檢索增強生成 (RAG Engine)
* **Embedding**: 使用 Google Generative AI Embeddings 進行文本向量化。
* **Splitter**: 採用遞迴字元切割法，平衡文本完整性與檢索精確度。
* **Context Window**: 針對 Gemini 模型特性優化，提供豐富的 Context Fragments 以獲取更佳的生成品質。

### 3. Web UI
* 透過 **Streamlit** 實作，支援文件即時載入、對話紀錄保存以及檢索來源顯示。

## 📂 專案結構
* `chatbot.py`: Streamlit 應用的入口程式與 UI 邏輯。
* `rag_engine.py`: 核心 RAG 邏輯，包含 Embedding 與檢索鏈。
* `data_processing.py`: 文件讀取、清洗與正規化模組。
* `bulid_data.py`: 初始化向量資料庫與資料導入。
