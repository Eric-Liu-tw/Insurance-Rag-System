import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 1. 載入環境變數
load_dotenv()
GEMINI_API = os.getenv("gemini_key")
persist_dir = os.getenv("persist_dir", "./chroma_db")

if not GEMINI_API:
    raise RuntimeError("請在 .env 中設定 gemini_key")

# 2. 初始化 Embedding 模型 (確保與建庫時一致)
embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key=GEMINI_API,
    model="models/text-embedding-004", # 依照您的要求使用 004 版本
)

# 3. 連接 Chroma 資料庫
vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings,
    collection_name="Travel_Insurance_RAG", # 必須與 build_data.py 一致
)

# 4. 取得資料庫統計
count = vectorstore._collection.count()
print(f"\n📊 資料庫統計：目前共有 {count} 個 Chunks")
print("="*80)

if count == 0:
    print("❌ 資料庫是空的！請先執行 build_data.py 進行建庫。")
else:
    # 5. 直接取得所有資料（包含 metadatas 和 documents）
    all_data = vectorstore._collection.get(include=["metadatas", "documents"])

    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    ids = all_data["ids"]

    # 6. 逐一列印檢查
    for i, (doc_text, meta, doc_id) in enumerate(zip(documents, metadatas, ids), 1):
        print(f"【Chunk {i}】 ID: {doc_id}")
        
        # 修正：對應 data_processing.py 中定義的 Key
        source = meta.get('source', '未知')
        a_no = meta.get('article_no', '無')          # 原本寫 article_nos 改為 article_no
        a_title = meta.get('article_title', '無')    # 原本寫 article_titles 改為 article_title

        print(f"📄 來源檔案: {source}")
        print(f"⚖️ 條號: {a_no}")
        print(f"📝 條文標題: {a_title}")
        print("-" * 40)
        
        # 內容預覽
        preview = doc_text.replace('\n', ' ').strip()
        print(f"📖 內容預覽：{preview[:150]}...")
        print("\n" + "="*80 + "\n")

print("✅ 檢查完成！")