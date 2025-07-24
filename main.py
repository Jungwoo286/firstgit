"""
Local RAG AI Assistant 메인 파이프라인
- 문서 임베딩, 검색, rerank, LLM 답변 생성 전체 흐름 제어
- CLI 기반 질문/응답
"""
import os
from embedder import Embedder
from retriever import Retriever
from reranker import Reranker
from llm_runner import LLMRunner
from prompt_template import get_system_prompt, build_structured_prompt
from utils.file_io import ensure_dir
import yaml


def load_config(config_path: str = 'config.yaml') -> dict:
    """
    config.yaml 파일을 로드합니다.
    Returns:
        dict: 설정 딕셔너리
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[Error] config.yaml 로드 실패: {e}")
        return {}

def setup_environment():
    """
    필수 폴더(db, models, dist) 자동 생성
    """
    ensure_dir('db')
    ensure_dir('models')
    ensure_dir('dist')

def main():
    """
    전체 파이프라인 실행 함수
    - CLI 입력 → 임베딩/검색/리랭크/LLM 답변 → 출력
    """
    # 1. 환경/설정 로드
    setup_environment()
    config = load_config()
    if not config:
        print('설정 파일을 불러올 수 없습니다. 종료합니다.')
        return

    # 2. 모듈 초기화
    embedder = Embedder(config['embedding_model_path'])
    retriever = Retriever(config['chroma_db_path'], config['embedding_model_path'])
    reranker = Reranker(config['reranker_model_path'])
    llm = LLMRunner(config['llm_model_path'])

    # 3. CLI 루프
    while True:
        question = input('\n질문을 입력하세요 (종료: exit): ').strip()
        if question.lower() in ['exit', 'quit', '종료']:
            print('종료합니다.')
            break
        try:
            # 4. 검색
            retrieved = retriever.query(question, config.get('retriever_top_k', 10))
            if not retrieved:
                print('문서에서 유사 문단을 찾을 수 없습니다.')
                continue
            passages = [x[0] for x in retrieved]

            # 5. rerank
            reranked = reranker.rerank(question, passages, config.get('reranker_top_n', 3))
            if not reranked or reranked[0][1] < 0.2:  # 점수 임계치 예시
                print('문서에서 충분한 정보를 찾을 수 없습니다.')
                continue
            top_contexts = [x[0] for x in reranked]

            # 6. 프롬프트 생성 및 LLM 답변
            system_prompt = get_system_prompt()
            prompt = build_structured_prompt(system_prompt, question, top_contexts)
            answer = llm.generate_answer(prompt)
            if not answer or len(answer.strip()) < 10:
                print('충분한 답변을 생성하지 못했습니다.')
            else:
                print(f'\n[답변]\n{answer.strip()}')
        except Exception as e:
            print(f'[Error] 파이프라인 실행 중 오류: {e}')

if __name__ == '__main__':
    main() 