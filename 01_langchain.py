# 01_langchain.py
# 목적: Playground에서 쓰던 System/Prompt/모델 설정을 그대로 LangChain으로 재현
# 사용법:
#   1) 아래 SYSTEM / USER_TEMPLATE에 Playground에서 쓰던 문구를 붙여 넣는다.
#   2) .env에 OPENAI_API_KEY=sk-... 가 있어야 한다.
#   3) python 01_langchain.py 로 실행 (아래 TEXT를 바꿔도 됨)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()  # .env에서 OPENAI_API_KEY 불러오기

# ─────────────────────────────────────────────────────────
# ① 여기에 Playground의 "System instructions"를 그대로 붙여넣으세요.
SYSTEM = """너는 문서를 요약하는 도우미야. 한국어로 3문장 이내의 핵심 요약만 제공해. 답변의 끝에는 이모지를 붙이고, 불확실한 내용은 제거하여 요약해줘. 말투는 격식있는 말투로 해.
"""
# ② 여기에 Playground의 "User" 메시지 템플릿을 붙여넣으세요.
#   (Playground에서 Variables를 썼다면 {변수명}으로 매핑)
USER_TEMPLATE = """요약해줘:
{text}"""
# ─────────────────────────────────────────────────────────

# ③ Playground에서 선택했던 모델/파라미터를 동일하게 맞추세요.
llm = ChatOpenAI(
    model="gpt-4o-mini",   # ← Playground에서 쓰던 모델명으로 교체 가능
    temperature=0.2        # ← Playground의 Temperature와 동일하게
)

# ④ System + User 메시지를 합쳐 프롬프트 구성
# zero-shot
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user", USER_TEMPLATE),
])

# ⑤ 실행 체인 구성 (프롬프트 → LLM)
chain = prompt | llm

if __name__ == "__main__":
    # 테스트용 본문 (원하는 텍스트로 교체 가능)
    TEXT = """내부 지식베이스의 검색 정확도를 높이기 위해 임베딩 모델을 교체하는 실험을 진행했다. 도메인 특화 용어가 많은 문서를 대상으로 500개의 검증 쿼리를 만들고, 기존 모델과 신규 모델의 검색 상위 5개 문서 재현율을 비교했다. 신규 모델은 재현율이 평균 8%p 개선되었지만 벡터 차원이 커져 인덱스 크기와 빌드 시간이 증가했고, 쿼리당 검색 지연도 평균 20ms 늘어났다. 비용·지연을 고려해, 일반 질의에는 기존 모델을 유지하고 OOD(도메인 밖)로 감지되는 질의에만 신규 모델을 사용하는 하이브리드 전략을 채택했다. 질의 분류기는 과거 로그로 학습한 경량 분류 모델을 사용해 초저지연으로 결정한다."""
    
    # 체인 실행 (Playground의 Variables에 대응: {"text": TEXT})
    result = chain.invoke({"text": TEXT})
    print(result.content)   # LLM 응답 본문
