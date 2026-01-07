import os
import shutil
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# 匯入我們優化過的處理函數
from data_processing import process_document, split_text_with_metadata

load_dotenv()

FILE_PATH = os.getenv("file_path")
PERSIST_DIR = os.getenv("persist_dir", "./chroma_db")
GEMINI_API = os.getenv("gemini_key")

def main():
    if not FILE_PATH or not GEMINI_API:
        raise ValueError("請檢查 .env 是否設定 file_path 與 gemini_key")

    # 1. 清理舊資料庫 (選擇性：如果要確保乾淨重建，建議開啟)
    if os.path.exists(PERSIST_DIR):
        print(f"正在清理舊資料庫: {PERSIST_DIR}...")
        shutil.rmtree(PERSIST_DIR)

    print(f"正在處理文件: {FILE_PATH}...")
    
    # 2. 讀取與切割
    text = process_document(FILE_PATH)
    source = os.path.basename(FILE_PATH)
    # 使用新的切割邏輯 (會保留更完整的標題上下文)
    chunks = split_text_with_metadata(text, source)

    print(f"文件切割完成，共產生 {len(chunks)} 個片段。")

    # 3. 轉換為 LangChain Document 物件
    documents = [
        Document(page_content=c["text"], metadata=c["metadata"])
        for c in chunks
    ]

    # 4. 向量化並儲存
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GEMINI_API
    )

    print("正在建立向量索引 (這可能需要一點時間)...")
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="Travel_Insurance_RAG",
    )
    print(f"✅ 建庫完成！資料已儲存至 {PERSIST_DIR}")

if __name__ == "__main__":
    main()