# 목적: 브라우저에서 대화하는 간단 UI
# 포인트:
#  - streamlit의 st.chat_input, st.chat_message로 챗 형태 UI 구성
#  - 세션 상태(st.session_state)에 (role, content) 튜플을 쌓아 간단히 히스토리 유지
#  - 매우 단순화를 위해 프롬프트에 히스토리를 "문자열"로 이어붙여 모델에 전달
#    (실전에서는 LangChain 메모리(RunnableWithMessageHistory)를 서버 사이드로 넣는 편이 더 안전)

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# 페이지 메타 설정(브라우저 탭 제목/아이콘)
st.set_page_config(page_title="LangChain Chat", page_icon="💬")

# ==========================
# [ADDED] 사이드바: 모델/온도 설정
# ==========================
with st.sidebar:
    st.subheader("설정")
    model = st.selectbox("모델", ["gpt-4o-mini", "gpt-4o"], index=0)
    temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    st.caption("모델과 샘플링 강도(Temperature) 설정")

# ==========================
# [MODIFIED] 사이드바 선택값으로 LLM 생성
# ==========================
llm = ChatOpenAI(model=model, temperature=temp)

# 세션 상태에 메시지 리스트가 없으면 초기화
if "messages" not in st.session_state:
    # ("user"|"assistant", content)
    st.session_state["messages"] = []

st.title("LangChain 과 OpenAI 를 활용한 간단한 챗봇 구현🤖 ")

# 과거 메시지를 UI에 렌더
for role, content in st.session_state["messages"]:
    with st.chat_message(role):
        st.markdown(content)

# 하단 입력창 (엔터로 전송)
user = st.chat_input("메시지를 입력하세요...")
if user:
    # 1) 사용자 메시지 추가/표시
    st.session_state["messages"].append(("user", user))
    with st.chat_message("user"):
        st.markdown(user)

    # 2) 간단한 히스토리 문자열을 만들어 모델에 전달
    #    (최근 6턴만 사용하여 프롬프트 길이 과도 증가를 방지)
    history_text = "\n".join([f"{r.upper()}: {c}" for r, c in st.session_state["messages"][-6:]])
    prompt = (
        "다음 대화 히스토리를 참고하여 마지막 사용자 메시지에 간결하게 한국어로 답하라.\n"
        f"{history_text}\n\n답변:"
    )

    # 3) 모델 호출
    answer = llm.invoke(prompt).content

    # 4) 어시스턴트 메시지 추가/표시
    st.session_state["messages"].append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)
