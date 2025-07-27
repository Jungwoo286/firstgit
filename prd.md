🧩 Product Requirements Document (PRD)
📌 프로젝트명
Local RAG AI Assistant using EEVE‑Korean‑Instruct‑2.8B‑v1.0 + ChromaDB + llama.cpp

🧭 목적
이 제품은 PDF, TXT, Markdown 등 비정형 학습 자료를 벡터화하여 로컬에 저장하고,
사용자가 자연어로 질문하면 해당 문서를 기반으로 정확한 답변을 생성하는 로컬 기반 AI 어시스턴트입니다.
회사에서 사용할것이며 회사의 전략기획팀을 위한 팀전용 소버린 AI 어시스턴트입니다 오프라인상황에서도 사용가능합니다

🎯 주요 기능 (Key Features)
1. 문서 임베딩
사용자가 제공한 문서를 Chunk 단위로 분할

고성능 임베딩 모델(BGE-small-en-v1.5, E5-base-v2 등)을 사용하여 문단 임베딩

ChromaDB에 벡터 저장 (Persistent 모드)

2. 질의 기반 유사 문단 검색 (Retriever)
사용자의 질문을 동일 임베딩 모델로 벡터화

ChromaDB에서 코사인 유사도로 Top-K 문단 검색 (기본 10개)

3. Reranker
검색된 Top-K 문단을 cross-encoder 기반 reranker (MiniLM 계열)로 재정렬

상위 1~3개의 문단만 LLM 입력으로 사용

4. LLM 답변 생성
EEVE‑Korean‑Instruct‑2.8B‑v1.0 (llama.cpp + GGUF 로컬 실행) 기반으로 프롬프트 템플릿에 context 삽입

System Prompt, 질문, 문단 순으로 구성된 structured prompt 사용

5. 경량화된 배포 패키지
llama-cpp-python 기반으로 최적화된 실행 환경 구성

로컬에서 인터넷 연결 없이 동작 (벡터 검색 및 추론 모두 오프라인)

🏗️ 시스템 구성도
```
[User Question]
      ↓
[Embedding: Question → Vector]
      ↓
[Retriever (ChromaDB)]
      ↓
[Top-10 문단 → Reranker (Cross-Encoder)]
      ↓
[Top-3 문단 → LLM Prompt]
      ↓
[EEVE‑Korean‑Instruct‑2.8B‑v1.0 (llama.cpp) → Answer]
      ↓
[Return Final Answer]
```

⚙️ 기술 스택
항목	기술
벡터DB	ChromaDB (Persistent mode)
임베딩 모델	BGE-small-en-v1.5 또는 intfloat/e5-base-v2
reranker	cross-encoder/ms-marco-MiniLM-L6-en-de-v1
LLM	EEVE‑Korean‑Instruct‑2.8B‑v1.0 (llama.cpp + GGUF)
프롬프트 엔진	Custom, with system-template 구조
패키징	llama-cpp-python + 경량화된 배포 패키지
인터페이스	CLI or Optional FastAPI UI
설정 관리	.env, config.yaml

🧪 테스트 시나리오
시나리오	기대 결과
질문: "퇴직금 계산은 어떻게 하나요?"	노동법 기준 정확한 계산 공식과 요건 설명
질문: "고객 개인정보는 어떻게 보호해야 하나요?"	회사 규정 or 법률 문단에서 인용된 설명 포함
질문: "올해 실적 요약 알려줘"	문서 내 KPI 요약 문단 기반으로 응답

🧱 코드 구성
```
project/
│
├── main.py                  # 전체 파이프라인 제어
├── embedder.py             # 문서 임베딩
├── retriever.py            # 유사도 기반 검색
├── reranker.py             # 문서 정렬
├── llm_runner.py           # LLM 프롬프트 + 응답 생성 (llama.cpp)
├── prompt_template.py      # 프롬프트 템플릿 관리
├── utils/                  # 공통 유틸 함수
│   └── file_io.py, path.py
│
├── config.yaml             # 경로/모델명 설정
├── .env                    # 민감 정보 또는 실행 변수
├── db/                     # ChromaDB 벡터 저장
├── models/                 # 임베딩, reranker, GGUF 모델 저장 위치
├── dist/                   # 배포 패키지 생성 위치
```

⚠️ 예외 처리 및 안정성 설계
ChromaDB가 없거나 index가 비어있을 경우 friendly error 출력

reranker 점수가 낮을 경우 답변 대신 "문서에서 충분한 정보를 찾을 수 없습니다." 출력

LLM 응답이 비어있거나 너무 짧을 경우 fallback 메시지 제공

모델 파일 경로가 없을 시 자동 생성 유도

🧰 개발 원칙 (Cursor Rules 반영)
모든 코드에 Docstring, 에러 방어 로직, 상대경로 처리 필수

설정 하드코딩 금지 → .env, config.yaml 필수화

한 함수는 하나의 책임만 가지도록 설계

코드를 작성한 후 내부적으로 에러 가능성과 구조적 개선 여부를 체크할 것

🚀 릴리즈 및 실행 방식
최종 사용자는 배포 패키지 실행만으로 실행 가능

CLI 상에서 질문 입력 → 답변 출력 형태

추후 FastAPI 또는 Electron 기반 GUI로 확장 고려

---

### EEVE‑Korean‑Instruct‑2.8B‑v1.0 GGUF 변환 및 llama.cpp 세팅 가이드

#### 1. GGUF 변환
```bash
# llama-cpp-python 설치
pip install llama-cpp-python

# GGUF 변환
python -m llama_cpp.convert_llama_weights_to_gguf ./EEVE-Korean-Instruct-2.8B-v1.0 --outfile ./models/eeve-korean-2.8b.gguf
```

#### 2. config.yaml 수정
```yaml
llm_model_path: './models/eeve-korean-2.8b.gguf'
```

#### 3. llm_runner.py 예시 코드
```python
from llama_cpp import Llama

class LLMRunner:
    """
    EEVE-Korean-Instruct-2.8B-v1.0 GGUF 모델을 llama.cpp로 로컬 추론하는 클래스
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llm = Llama(
            model_path=model_path, 
            n_ctx=2048,
            n_threads=4,  # CPU 스레드 수 조정
            n_gpu_layers=0  # GPU 사용 시 1 이상으로 설정
        )

    def generate_answer(self, prompt: str) -> str:
        try:
            output = self.llm(
                prompt, 
                max_tokens=512, 
                stop=["</s>", "\n\n"],
                temperature=0.7
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            print(f"[Error] LLM 추론 실패: {e}")
            return ""
```

---

### 배포 패키지 구성
```
rag-app-portable/
├── llama-cpp-python/       # 최적화된 llama-cpp-python
├── models/                 # GGUF 모델 + 임베딩 모델
├── app/                    # 애플리케이션 코드
├── run.bat                 # Windows 실행
├── run.sh                  # Linux/Mac 실행
└── README.txt              # 사용법
```

**예상 배포 크기: ~1.6GB** (기존 7GB 대비 77% 감소)

---

**모델 변환이 끝나면 "완료"라고 답해 주세요.**  
(이후 end-to-end 파이프라인 테스트, 추가 코드 보완 등 바로 도와드릴 수 있습니다!)

