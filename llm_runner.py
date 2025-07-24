"""
LLM 프롬프트 및 응답 생성 모듈
- 프롬프트 구성
- LLM 호출 및 답변 생성
"""
from typing import List

class LLMRunner:
    """
    LLM을 통해 답변을 생성하는 클래스
    """
    def __init__(self, model_path: str):
        """
        Args:
            model_path (str): LLM 모델 경로
        """
        pass

    def build_prompt(self, system_prompt: str, question: str, contexts: List[str]) -> str:
        """
        프롬프트 템플릿에 context/질문을 삽입
        Args:
            system_prompt (str): 시스템 프롬프트
            question (str): 사용자 질문
            contexts (List[str]): 문단 리스트
        Returns:
            str: 완성된 프롬프트
        """
        pass

    def generate_answer(self, prompt: str) -> str:
        """
        LLM에 프롬프트를 입력해 답변 생성
        Args:
            prompt (str): LLM 입력 프롬프트
        Returns:
            str: 생성된 답변
        """
        pass 