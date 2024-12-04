import streamlit as st

from langchain.memory import ConversationBufferMemory
from utils import qa_agent

st.title("💊 건강 보조품 추천 시스템")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API 키를 입력하세요：", type="password")
    st.markdown("[OpenAI API key 가져오세요](https://platform.openai.com/account/api-keys)")

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

uploaded_file = st.file_uploader("PDF 파일을 업로드해 주세요: ", type="pdf")
question = st.text_input("증상을 입력하세요:", disabled=not uploaded_file)

if uploaded_file and question and not openai_api_key:
    st.info("OpenAI API 키를 입력하세요")

if uploaded_file and question and openai_api_key:
    with st.spinner("AI가 생각 중입니다. 잠시만 기다려 주세요..."):
        response = qa_agent(openai_api_key, st.session_state["memory"], uploaded_file, question)

    if "error" in response:
        st.error(response["error"])
    else:
        st.write("### 대답")
        st.write(response["answer"])
    st.session_state["chat_history"] = response["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("역사 소식"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i + 1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()
