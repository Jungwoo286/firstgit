"""
문서 임베딩 모듈
- 문서 chunk 분할
- 임베딩 모델 로드 및 임베딩 생성
- 벡터 DB 저장
"""
import os
from typing import List

class Embedder:
    """
    임베딩 모델을 로드하고, 문서/문단을 임베딩 벡터로 변환하는 클래스
    """
    def __init__(self, model_path: str):
        """
        Args:
            model_path (str): 임베딩 모델 경로
        Raises:
            FileNotFoundError: 모델 파일이 없을 경우
        """
        pass

    def chunk_document(self, text: str, chunk_size: int = 512) -> List[str]:
        """
        문서를 chunk 단위로 분할
        Args:
            text (str): 전체 문서 텍스트
            chunk_size (int): chunk 크기
        Returns:
            List[str]: 분할된 문단 리스트
        """
        pass

    def embed_texts(self, texts: List[str]) -> List[list]:
        """
        여러 문단을 임베딩 벡터로 변환
        Args:
            texts (List[str]): 문단 리스트
        Returns:
            List[list]: 임베딩 벡터 리스트
        """
        pass 