import streamlit as st
import os
from dotenv import load_dotenv
from rag_engine import InsuranceRagEngine

# 1. 初始化環境
load_dotenv()
st.set_page_config(page_title="旅遊不便險 AI 助手", layout="wide")

# 2. 初始化引擎
@st.cache_resource
def get_engine():
    return InsuranceRagEngine()

try:
    engine = get_engine()
except ValueError as e:
    st.error(f"系統初始化失敗: {e}")
    st.stop()

# 3. Session State 管理
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. 顯示標題與歷史訊息
st.title("旅遊不便險 AI 助手 🤖")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. 處理使用者輸入
if prompt := st.chat_input("請輸入您的問題..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("正在查閱條款並分析..."):
            # --- 關鍵修正：僅傳入當前問題，不傳入對話歷史 ---
            result = engine.get_answer(prompt) 
            response = result["answer"]
            
            st.markdown(response)

            # 顯示參考來源
            if result["source_documents"]:
                with st.expander("查看參考條款原文"):
                    for doc in result["source_documents"]:
                        meta = doc.metadata
                        # 同時支援多種 metadata key 名稱以防遺漏
                        a_no = meta.get('article_no', meta.get('section_id', '未知條號'))
                        a_title = meta.get('article_title', meta.get('title', ''))
                        
                        st.markdown(f"**{a_no} {a_title}**")
                        st.text(doc.page_content)
                        st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response})