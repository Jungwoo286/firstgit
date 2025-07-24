"""
유사도 기반 검색 모듈
- 질의 임베딩
- ChromaDB에서 유사 문단 검색
"""
from typing import List, Tuple

class Retriever:
    """
    ChromaDB에서 코사인 유사도로 Top-K 문단을 검색하는 클래스
    """
    def __init__(self, db_path: str, embedding_model_path: str):
        """
        Args:
            db_path (str): ChromaDB 경로
            embedding_model_path (str): 임베딩 모델 경로
        """
        pass

    def query(self, question: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        질의를 임베딩 후, 유사 문단 Top-K 검색
        Args:
            question (str): 사용자 질문
            top_k (int): 반환할 문단 수
        Returns:
            List[Tuple[str, float]]: (문단, 유사도 점수) 리스트
        """
        pass 