import os
import re
import docx
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 寬鬆版 Regex：允許條號前有換行，允許標題在下一行
ARTICLE_PATTERN = re.compile(r"(第\s*[一二三四五六七八九十百]+\s*條)\s+(.*)")

def process_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if ext == ".docx":
        doc = docx.Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs)
    elif ext == ".pdf":
        reader = PdfReader(file_path)
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
    else:
        raise ValueError("不支援的格式")
    
    # --- 新增：資料清洗 (非常重要) ---
    # 許多 PDF 轉出後，條號中間會有空白，例如 "第 三 十 九 條"
    # 這會導致 Regex 失敗。我們要先把這些空白殺掉。
    def normalize_article_no(match):
        return match.group(0).replace(" ", "")
    
    # 尋找 "第 X 條" 的模式並移除其中的空白
    text = re.sub(r"第\s*[一二三四五六七八九十百\s]+\s*條", normalize_article_no, text)
    
    return text

def split_text_with_metadata(text, source_name, chunk_size=800, overlap=150):
    # 嘗試抓取條文
    matches = list(re.finditer(ARTICLE_PATTERN, text))
    results = []
    
    # --- 新增：Fallback 機制 ---
    # 如果 Regex 什麼都沒抓到 (matches 為空)，不要讓程式跑完卻沒資料
    # 直接使用純文字切割，至少讓資料庫有東西
    if not matches:
        print("⚠️ 警告：Regex 未匹配到任何條文結構，切換至一般文字切割模式。")
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = splitter.split_text(text)
        for chunk in chunks:
            results.append({
                "text": chunk,
                "metadata": {
                    "source": source_name,
                    "article_no": "Unknown", 
                    "article_title": "General Context",
                    "chunk_id": "fallback"
                }
            })
        return results

    # 如果有抓到條文，繼續執行精準切割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "；", " ", ""]
    )

    for i, match in enumerate(matches):
        article_no = match.group(1)
        article_title = match.group(2).strip()
        
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        
        full_text = f"條款：{article_no} {article_title}\n內容：{content}"

        if len(full_text) > chunk_size:
            sub_chunks = splitter.split_text(full_text)
            for idx, sub in enumerate(sub_chunks):
                context_text = f"【續】{article_no} {article_title}\n{sub}"
                results.append({
                    "text": context_text,
                    "metadata": {
                        "source": source_name,
                        "article_no": article_no,
                        "article_title": article_title,
                        "chunk_id": f"{article_no}_{idx}"
                    }
                })
        else:
            results.append({
                "text": full_text,
                "metadata": {
                    "source": source_name,
                    "article_no": article_no,
                    "article_title": article_title,
                    "chunk_id": article_no
                }
            })
            
    return results