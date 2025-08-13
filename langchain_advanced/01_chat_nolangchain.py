# 목적: LangChain 없이도 가능한 가장 단순한 "한 번 호출" 흐름을 경험
# 포인트:
#  - load_dotenv(): .env에서 OPENAI_API_KEY 읽어오기
#  - ChatOpenAI: OpenAI 채팅 모델을 파이썬 객체로 다룰 수 있게 해주는 래퍼
#  - llm.invoke("질문"): 한 번 묻고 한 번 답 받기

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# .env 파일에서 환경변수(OPENAI_API_KEY 등)를 불러옵니다.
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# 한 번 질문해 보기 (단일 턴)
question = "안녕! 너는 무엇을 할 수 있어?"
# invoke()는 모델에 메시지를 보내고, AIMessage 객체를 반환
resp = llm.invoke(question)

# 실제 텍스트는 resp.content에 들어있음
print(resp.content)
