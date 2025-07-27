"""
Retriever Module
검색된 문서들을 재정렬하고 필터링하는 모듈
"""

import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import yaml

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentRetriever:
    """
    검색된 문서들을 재정렬하고 필터링하는 클래스
    """
    
    def __init__(self, reranker_model_path: str, top_k: int = 10, top_n: int = 3):
        """
        Document Retriever 초기화
        
        Args:
            reranker_model_path (str): Reranker 모델 경로
            top_k (int): 초기 검색 결과 수
            top_n (int): 최종 선택할 문서 수
        """
        self.reranker_model_path = reranker_model_path
        self.top_k = top_k
        self.top_n = top_n
        
        # Reranker 모델 로딩
        logger.info(f"Reranker 모델 로딩 중: {reranker_model_path}")
        try:
            self.reranker = CrossEncoder(reranker_model_path)
            logger.info("Reranker 모델 로딩 완료")
        except Exception as e:
            logger.error(f"Reranker 모델 로딩 실패: {e}")
            # Reranker 없이도 동작하도록 설정
            self.reranker = None
            logger.warning("Reranker 없이 진행합니다.")
    
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        검색된 문서들을 reranker로 재정렬
        
        Args:
            query (str): 사용자 질문
            documents (List[Dict]): 검색된 문서들
            
        Returns:
            List[Dict]: 재정렬된 문서들
        """
        if not documents:
            return []
        
        if self.reranker is None:
            # Reranker가 없으면 거리 기반으로 정렬
            logger.warning("Reranker가 없어 거리 기반으로 정렬합니다.")
            sorted_docs = sorted(documents, key=lambda x: x['distance'])
            return sorted_docs[:self.top_n]
        
        try:
            # 쿼리-문서 쌍 생성
            pairs = []
            for doc in documents:
                pairs.append([query, doc['document']])
            
            # Reranker 점수 계산
            logger.info(f"{len(pairs)}개 문서 reranking 중...")
            scores = self.reranker.predict(pairs)
            
            # 점수와 문서 결합
            scored_docs = []
            for i, doc in enumerate(documents):
                scored_doc = doc.copy()
                scored_doc['reranker_score'] = float(scores[i])
                scored_docs.append(scored_doc)
            
            # 점수 기준으로 정렬 (높은 점수가 위로)
            sorted_docs = sorted(scored_docs, key=lambda x: x['reranker_score'], reverse=True)
            
            # 상위 N개 선택
            top_docs = sorted_docs[:self.top_n]
            
            logger.info(f"Reranking 완료: {len(top_docs)}개 문서 선택")
            return top_docs
            
        except Exception as e:
            logger.error(f"Reranking 실패: {e}")
            # 오류 시 거리 기반으로 정렬
            sorted_docs = sorted(documents, key=lambda x: x['distance'])
            return sorted_docs[:self.top_n]
    
    def filter_documents(self, documents: List[Dict[str, Any]], min_score: float = 0.5) -> List[Dict[str, Any]]:
        """
        낮은 점수의 문서 필터링
        
        Args:
            documents (List[Dict]): 문서들
            min_score (float): 최소 점수 임계값
            
        Returns:
            List[Dict]: 필터링된 문서들
        """
        if not documents:
            return []
        
        filtered_docs = []
        for doc in documents:
            # reranker 점수가 있으면 사용, 없으면 거리 기반 점수 사용
            if 'reranker_score' in doc:
                score = doc['reranker_score']
            else:
                # 거리를 점수로 변환 (거리가 작을수록 높은 점수)
                score = 1.0 - doc['distance']
            
            if score >= min_score:
                filtered_docs.append(doc)
        
        logger.info(f"필터링 완료: {len(filtered_docs)}/{len(documents)}개 문서 유지")
        return filtered_docs
    
    def get_context_texts(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        문서들에서 텍스트만 추출
        
        Args:
            documents (List[Dict]): 문서들
            
        Returns:
            List[str]: 텍스트 리스트
        """
        return [doc['document'] for doc in documents]
    
    def process_search_results(self, query: str, search_results: List[Dict[str, Any]]) -> List[str]:
        """
        검색 결과를 처리하여 최종 컨텍스트 생성
        
        Args:
            query (str): 사용자 질문
            search_results (List[Dict]): 검색 결과
            
        Returns:
            List[str]: 최종 컨텍스트 텍스트들
        """
        if not search_results:
            logger.warning("검색 결과가 없습니다.")
            return []
        
        # 상위 K개 선택
        top_k_results = search_results[:self.top_k]
        
        # Reranking
        reranked_results = self.rerank_documents(query, top_k_results)
        
        # 필터링
        filtered_results = self.filter_documents(reranked_results)
        
        # 텍스트 추출
        context_texts = self.get_context_texts(filtered_results)
        
        logger.info(f"최종 컨텍스트: {len(context_texts)}개 문단")
        return context_texts


def create_retriever(config_path: str = "config.yaml") -> DocumentRetriever:
    """
    Document Retriever 인스턴스 생성
    
    Args:
        config_path (str): 설정 파일 경로
        
    Returns:
        DocumentRetriever: Document Retriever 인스턴스
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return DocumentRetriever(
        reranker_model_path=config['reranker_model_path'],
        top_k=config['retriever_top_k'],
        top_n=config['reranker_top_n']
    )


if __name__ == "__main__":
    # 테스트 코드
    retriever = create_retriever()
    
    # 테스트 데이터
    test_query = "인공지능이란 무엇인가요?"
    test_documents = [
        {
            'document': '인공지능(AI)은 인간의 학습능력과 추론능력을 인공적으로 구현한 시스템입니다.',
            'distance': 0.1,
            'metadata': {'source': 'test1.txt'}
        },
        {
            'document': '머신러닝은 데이터로부터 패턴을 학습하는 기술입니다.',
            'distance': 0.3,
            'metadata': {'source': 'test2.txt'}
        },
        {
            'document': '딥러닝은 신경망을 사용한 머신러닝의 한 분야입니다.',
            'distance': 0.2,
            'metadata': {'source': 'test3.txt'}
        }
    ]
    
    # 처리 테스트
    context_texts = retriever.process_search_results(test_query, test_documents)
    
    print("최종 컨텍스트:")
    for i, text in enumerate(context_texts):
        print(f"{i+1}. {text}") 