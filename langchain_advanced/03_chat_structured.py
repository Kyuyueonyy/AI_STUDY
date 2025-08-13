# 목적: LLM의 답을 "항상 일정한 형태(JSON)"로 받기
#  - ResponseSchema(name=..., description=...): 결과에 어떤 필드가 있어야 하는지 정의
#  - StructuredOutputParser: LLM에게 "이 JSON 형식을 지키라"는 지시문을 만들고, 응답을 dict로 파싱
#  - format_instructions: 위 파서가 자동 생성하는 "형식 지시문" 문자열(프롬프트에 넣어야 파싱 성공 확률 ↑)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 최신 권장 경로
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# 1) JSON 스키마 정의: 어떤 필드를 어떤 의미로 받을지 라벨링
schemas = [
    ResponseSchema(name="answer", description="간결한 한두 문장 답변"),
    ResponseSchema(name="notes", description="추가 메모(선택, 없으면 빈 문자열)"),
]

# 2) 스키마로부터 파서 생성 + 형식 지시문 얻기
parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()

# 3) 프롬프트: system(규칙) + user(질문 + 형식 지시문)
prompt = ChatPromptTemplate.from_messages([
    ("system", "항상 지정된 JSON 형식으로만 답하라. 여분의 텍스트를 쓰지 마라."),
    ("user", "질문: {q}\n형식 지시문:\n{fmt}")
])

# 4) LCEL 파이프라인: 프롬프트 → LLM → 파서
chain = prompt | llm | parser

if __name__ == "__main__":
    q = "RAG를 한 문장으로 설명해줘."
    # format_instructions를 프롬프트 변수로 함께 전달해야 모델이 JSON으로 답하려고 함
    result = chain.invoke({"q": q, "fmt": format_instructions})

    # result는 파싱된 dict
    print(result)            # 예: {'answer': '...', 'notes': '...'}
    print(result["answer"])  # 필드만 꺼내 쓰기
