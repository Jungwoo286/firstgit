🧩 Product Requirements Document (PRD)
📌 프로젝트명
Local RAG AI Assistant using Qwen 2.5 3B + ChromaDB

🧭 목적
이 제품은 PDF, TXT, Markdown 등 비정형 학습 자료를 벡터화하여 로컬에 저장하고,
사용자가 자연어로 질문하면 해당 문서를 기반으로 정확한 답변을 생성하는 로컬 기반 AI 어시스턴트입니다.

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
KULLM 1 3B (로컬 실행) 기반으로 프롬프트 템플릿에 context 삽입

System Prompt, 질문, 문단 순으로 구성된 structured prompt 사용

5. 실행파일화 및 배포
PyInstaller 또는 Nuitka 기반으로 전체 시스템을 .exe 실행파일로 패키징

로컬에서 인터넷 연결 없이 동작 (벡터 검색 및 추론 모두 오프라인)

🏗️ 시스템 구성도
csharp
복사
편집
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
[KULLM 1 3B → Answer]
      ↓
[Return Final Answer]
⚙️ 기술 스택
항목	기술
벡터DB	ChromaDB (Persistent mode)
임베딩 모델	BGE-small-en-v1.5 또는 intfloat/e5-base-v2
reranker	cross-encoder/ms-marco-MiniLM-L6-en-de-v1
LLM	KULLM 1 3B (Ollama 또는 직접 로컬 실행)
프롬프트 엔진	Custom, with system-template 구조
패키징	PyInstaller (Windows), dist/에 .exe 생성
인터페이스	CLI or Optional FastAPI UI
설정 관리	.env, config.yaml

🧪 테스트 시나리오
시나리오	기대 결과
질문: "퇴직금 계산은 어떻게 하나요?"	노동법 기준 정확한 계산 공식과 요건 설명
질문: "고객 개인정보는 어떻게 보호해야 하나요?"	회사 규정 or 법률 문단에서 인용된 설명 포함
질문: "올해 실적 요약 알려줘"	문서 내 KPI 요약 문단 기반으로 응답

🧱 코드 구성
bash
복사
편집
project/
│
├── main.py                  # 전체 파이프라인 제어
├── embedder.py             # 문서 임베딩
├── retriever.py            # 유사도 기반 검색
├── reranker.py             # 문서 정렬
├── llm_runner.py           # LLM 프롬프트 + 응답 생성
├── prompt_template.py      # 프롬프트 템플릿 관리
├── utils/                  # 공통 유틸 함수
│   └── file_io.py, path.py
│
├── config.yaml             # 경로/모델명 설정
├── .env                    # 민감 정보 또는 실행 변수
├── db/                     # ChromaDB 벡터 저장
├── models/                 # 임베딩 및 reranker 모델 저장 위치
├── dist/                   # 실행파일(.exe) 생성 위치
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
최종 사용자는 .exe 실행만으로 실행 가능

CLI 상에서 질문 입력 → 답변 출력 형태

추후 FastAPI 또는 Electron 기반 GUI로 확장 고려

