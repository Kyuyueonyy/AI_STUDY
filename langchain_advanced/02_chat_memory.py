#  같은 사용자가 이어서 대화하면 이전 맥락을 기억해서 답하도록 만들기

#  - ChatPromptTemplate: system/user 역할로 프롬프트를 구조화
#  - LCEL( | ): prompt → llm → parser(문자열) 파이프라인
#  - RunnableWithMessageHistory: 아무 체인에나 "대화 히스토리"를 덧씌우는 래퍼
# 흐름:
#  사용자가 한 턴 말할 때마다 → 히스토리 저장 → 다음 턴 입력 시 히스토리를 프롬프트에 주입

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# system: 항상 지켜야 할 규칙/톤/출력 정책
SYSTEM = (
    "너는 한국어로 답하는 친절한 챗봇이야. "
    "대화 맥락을 참고하되, 불확실하면 모른다고 말해해. 귀여운 말투를 사용하고, 대답의 끝부분에는 항상 이모지를 붙여줘."
)

# user: 이번 턴의 실제 입력(매번 달라짐)
USER = "{text}"

# 역할별 메시지로 프롬프트 구성
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user", USER),
])

# LLM 준비 + 문자열 파서(최종 응답을 string으로 받기)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
chain = prompt | llm | StrOutputParser()

# 세션별로 대화 기록을 저장할 딕셔너리(메모리 저장소)
store = {}

# 세션 ID를 키로 사용하여 히스토리 객체를 반환(없으면 새로 만듦)
def get_history(session_id: str):
    return store.setdefault(session_id, ChatMessageHistory())

# 기존 체인(chain)에 "메시지 히스토리" 기능을 감싸서 붙입니다.
# - input_messages_key="text": 이번 호출에서 사용자가 넣는 입력 키 이름
# - history_messages_key="history": 히스토리가 내부적으로 주입될 키(대부분 기본값 사용)
chat = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="text",
    history_messages_key="history",
)

if __name__ == "__main__":
    # 같은 사용자를 같은 세션으로 인식시키기 위한 설정
    cfg = {"configurable": {"session_id": "local_user"}}

    print("종료하려면 /quit 입력")
    while True:
        user = input("\n나: ").strip()
        if user == "/quit":
            break

        # chat.invoke() 시 해당 세션의 히스토리까지 자동으로 주입됨
        answer = chat.invoke({"text": user}, config=cfg)
        print("봇:", answer)
