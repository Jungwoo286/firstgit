"""
문서 정렬(Reranker) 모듈
- cross-encoder 기반 rerank
"""
from typing import List, Tuple

class Reranker:
    """
    cross-encoder 기반으로 문단을 재정렬하는 클래스
    """
    def __init__(self, model_path: str):
        """
        Args:
            model_path (str): reranker 모델 경로
        """
        pass

    def rerank(self, question: str, passages: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Top-K 문단을 rerank하여 상위 N개만 반환
        Args:
            question (str): 사용자 질문
            passages (List[str]): 검색된 문단 리스트
            top_n (int): 반환할 문단 수
        Returns:
            List[Tuple[str, float]]: (문단, rerank 점수) 리스트
        """
        pass 