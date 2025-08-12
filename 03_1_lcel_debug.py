from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda

## LCEL 파이프라인의 각 단계에서 실제로 오가는 데이터를 콘솔에 찍어보는 디버그용 예시시

load_dotenv()

SYSTEM = "너는 문서 요약 도우미다. 한국어로 2~3문장으로만 요약하라."
USER = "요약해줘:\n{text}"

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user", USER),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
parser = StrOutputParser()  # LLM 응답을 '그대로 문자열'로 받는 파서

# ─────────────────────────────────────────────────────────
# tap: 체인 중간값을 출력하고 그대로 다음 단계로 넘겨주는 "탭" 노드
def tap(tag: str):
    def _tap(x):
        print(f"\n===== {tag} =====")
        try:
            # 1) ChatPromptTemplate 출력: 메시지 리스트(BaseMessage들)
            from langchain.schema import BaseMessage
            if isinstance(x, list) and x and isinstance(x[0], BaseMessage):
                for i, m in enumerate(x, 1):
                    role = getattr(m, "type", type(m).__name__)
                    print(f"[{i}] role={role}")
                    print(m.content)
                    print("-" * 40)
                return x

            # 2) LLM 응답: AIMessage 객체(주로 .content에 텍스트가 들어 있음)
            content = getattr(x, "content", None)
            if content is not None:
                print(f"{type(x).__name__} → .content:")
                print(content)
                return x

            # 3) 파서 결과: 문자열/딕셔너리 등 최종 값
            print(repr(x))
            return x
        except Exception as e:
            print(f"(tap 에러) {e}")
            return x
    return RunnableLambda(_tap)
# ─────────────────────────────────────────────────────────

# LCEL 파이프라인에 "탭"을 끼워넣어 단계별로 출력
chain = (
    prompt
    | tap("1) 프롬프트 렌더 결과 (messages)")
    | llm
    | tap("2) LLM 원시 응답 객체 (AIMessage)")
    | parser
    | tap("3) 파서 결과 (최종 출력)")
)

if __name__ == "__main__":
    text = """내부 지식베이스의 검색 정확도를 높이기 위해 임베딩 모델을 교체하는 실험을 진행했다.
도메인 특화 용어가 많은 문서를 대상으로 500개의 검증 쿼리를 만들고, 기존 모델과 신규 모델의 검색 상위 5개 문서 재현율을 비교했다.
신규 모델은 재현율이 평균 8%p 개선되었지만 벡터 차원이 커져 인덱스 크기와 빌드 시간이 증가했고, 쿼리당 검색 지연도 평균 20ms 늘어났다.
비용·지연을 고려해, 일반 질의에는 기존 모델을 유지하고 OOD(도메인 밖)로 감지되는 질의에만 신규 모델을 사용하는 하이브리드 전략을 채택했다.
질의 분류기는 과거 로그로 학습한 경량 분류 모델을 사용해 초저지연으로 결정한다."""
    chain.invoke({"text": text})
