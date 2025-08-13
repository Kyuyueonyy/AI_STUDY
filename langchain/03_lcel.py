from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

load_dotenv()

SYSTEM = "너는 문서를 요약하는 도우미야. 한국어로 3문장 이내의 핵심 요약만 제공해. 답변의 끝에는 이모지를 붙이고, 불확실한 내용은 제거하여 요약해줘. 말투는 격식있는 말투로 해."
USER = "요약해줘:\n{text}"

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user", USER),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
parser = StrOutputParser()  # LLM 응답을 '그대로 문자열'로 받는 파서

# 핵심: 프롬프트 → LLM → 파서 파이프라인
# 파이프라인 한 줄로 처리
chain = prompt | llm | parser
# 이 프롬프트를 llm에 넣고, 나온 결과를 파서로 넘기는 파이프라인을 만든 것\
# 1. 프롬프트 변환
# 2. LLM 호출
# 3. 파싱

# prompt
# : ChatPromptTemplate 객체, 입력 변수를 채워서 LLM이 이해할 수 있는 "메시지 리스트"로 변환

# llm
# : ChatOpenAI 객체, 메시지를 받아서 LLM에 요청 후 결과(AIMessage) 반환

# parser
# : StrOutputParser 객체, AIMessage에서 .content 부분만 추출

if __name__ == "__main__":
    text = """내부 지식베이스의 검색 정확도를 높이기 위해 임베딩 모델을 교체하는 실험을 진행했다. 도메인 특화 용어가 많은 문서를 대상으로 500개의 검증 쿼리를 만들고, 기존 모델과 신규 모델의 검색 상위 5개 문서 재현율을 비교했다. 신규 모델은 재현율이 평균 8%p 개선되었지만 벡터 차원이 커져 인덱스 크기와 빌드 시간이 증가했고, 쿼리당 검색 지연도 평균 20ms 늘어났다. 비용·지연을 고려해, 일반 질의에는 기존 모델을 유지하고 OOD(도메인 밖)로 감지되는 질의에만 신규 모델을 사용하는 하이브리드 전략을 채택했다. 질의 분류기는 과거 로그로 학습한 경량 분류 모델을 사용해 초저지연으로 결정한다."""
    print(chain.invoke({"text": text}))
