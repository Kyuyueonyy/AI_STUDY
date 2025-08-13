import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 최신 LangChain 경로들
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

# 히스토리 구현 (in-memory)
from langchain_community.chat_message_histories import ChatMessageHistory
load_dotenv()
st.set_page_config(page_title="LangChain Chat (Memory)", page_icon="🧠")
st.title("LangChain + RunnableWithMessageHistory 챗봇")

# 1) 세션ID 준비: 사용자별로 대화기록을 분리하기 위한 식별자
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# 2) 서버 사이드 메모리 저장소(데모: 파이썬 dict)
#    - 운영에선 Redis/DB로 교체 가능
SERVER_STORE = st.session_state.get("_SERVER_STORE", {})
st.session_state["_SERVER_STORE"] = SERVER_STORE  # 새로고침 시 유지

def get_history(session_id: str):
    # 세션ID별로 ChatMessageHistory 객체를 보관
    return SERVER_STORE.setdefault(session_id, ChatMessageHistory())

# 3) 프롬프트(역할 분리)
SYSTEM = (
    "너는 한국어로 답하는 친절한 챗봇이다. "
    "대화 맥락을 참고하되, 사실에 자신 없으면 모른다고 답하라."
)
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user", "{text}")
])

# 4) 모델 + 파이프라인 (프롬프트 → LLM → 문자열 파서)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
base_chain = prompt | llm | StrOutputParser()

# 5) 아무 체인에나 히스토리를 “래핑”해서 붙이는 RunnableWithMessageHistory
chat_chain = RunnableWithMessageHistory(
    base_chain,
    get_history,                 # 세션별 기록을 가져오는 함수
    input_messages_key="text",   # 이번 턴 사용자 입력이 들어갈 키
    history_messages_key="history"  # 내부적으로 히스토리 주입에 쓰이는 키
)

# 6) UI 구성
with st.sidebar:
    st.caption(f"Session ID: `{st.session_state['session_id']}`")
    if st.button("대화 초기화"):
        # 서버 저장소와 화면 메시지 모두 초기화
        sid = st.session_state["session_id"]
        SERVER_STORE[sid] = ChatMessageHistory()
        st.session_state["messages"] = []
        st.rerun()

# 화면 표시용(선택): 과거 메시지 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 과거 메시지 렌더
for role, content in st.session_state["messages"]:
    with st.chat_message(role):
        st.markdown(content)

# 입력창
user = st.chat_input("메시지를 입력하세요...")
if user:
    # (화면 표시용) 사용자 메시지 추가
    st.session_state["messages"].append(("user", user))
    with st.chat_message("user"):
        st.markdown(user)

    # 7) 체인 실행: config에 session_id를 넘겨 히스토리를 자동 주입
    cfg = {"configurable": {"session_id": st.session_state["session_id"]}}
    answer = chat_chain.invoke({"text": user}, config=cfg)

    # (화면 표시용) 봇 메시지 추가
    st.session_state["messages"].append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)