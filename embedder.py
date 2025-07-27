"""
Document Embedder Module
문서를 임베딩하고 ChromaDB에 저장하는 모듈
"""

import os
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import yaml

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """
    문서를 임베딩하고 ChromaDB에 저장하는 클래스
    """
    
    def __init__(self, embedding_model_path: str, chroma_db_path: str):
        """
        Document Embedder 초기화
        
        Args:
            embedding_model_path (str): 임베딩 모델 경로
            chroma_db_path (str): ChromaDB 저장 경로
        """
        self.embedding_model_path = embedding_model_path
        self.chroma_db_path = chroma_db_path
        
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 임베딩 모델 로딩
        logger.info(f"임베딩 모델 로딩 중: {embedding_model_path}")
        try:
            self.embedding_model = SentenceTransformer(embedding_model_path)
            logger.info("임베딩 모델 로딩 완료")
        except Exception as e:
            logger.error(f"임베딩 모델 로딩 실패: {e}")
            raise
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        텍스트를 청크 단위로 분할
        
        Args:
            text (str): 분할할 텍스트
            chunk_size (int): 청크 크기
            overlap (int): 오버랩 크기
            
        Returns:
            List[str]: 분할된 텍스트 청크들
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 문장 경계에서 자르기
            if end < len(text):
                # 마지막 마침표나 줄바꿈을 찾아서 자르기
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                cut_point = max(last_period, last_newline)
                
                if cut_point > start:
                    end = cut_point + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트들을 임베딩
        
        Args:
            texts (List[str]): 임베딩할 텍스트들
            
        Returns:
            List[List[float]]: 임베딩 벡터들
        """
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]], collection_name: str = "documents") -> None:
        """
        문서들을 ChromaDB에 추가
        
        Args:
            documents (List[Dict]): 문서 리스트 (각 문서는 'content', 'metadata' 키 포함)
            collection_name (str): 컬렉션 이름
        """
        try:
            # 컬렉션 가져오기 또는 생성
            collection = self.client.get_or_create_collection(name=collection_name)
            
            all_chunks = []
            all_metadatas = []
            all_ids = []
            
            for i, doc in enumerate(documents):
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                
                # 텍스트 청킹
                chunks = self.chunk_text(content)
                
                for j, chunk in enumerate(chunks):
                    chunk_id = f"doc_{i}_chunk_{j}"
                    chunk_metadata = metadata.copy()
                    chunk_metadata['chunk_index'] = j
                    chunk_metadata['total_chunks'] = len(chunks)
                    
                    all_chunks.append(chunk)
                    all_metadatas.append(chunk_metadata)
                    all_ids.append(chunk_id)
            
            if all_chunks:
                # 임베딩 생성
                logger.info(f"{len(all_chunks)}개 청크 임베딩 중...")
                embeddings = self.embed_texts(all_chunks)
                
                # ChromaDB에 추가
                collection.add(
                    embeddings=embeddings,
                    documents=all_chunks,
                    metadatas=all_metadatas,
                    ids=all_ids
                )
                
                logger.info(f"{len(all_chunks)}개 청크가 ChromaDB에 저장되었습니다.")
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            raise
    
    def search_similar(self, query: str, n_results: int = 10, collection_name: str = "documents") -> List[Dict[str, Any]]:
        """
        유사한 문서 검색
        
        Args:
            query (str): 검색 쿼리
            n_results (int): 반환할 결과 수
            collection_name (str): 컬렉션 이름
            
        Returns:
            List[Dict]: 검색 결과 (문서, 메타데이터, 유사도 점수)
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            
            # 쿼리 임베딩
            query_embedding = self.embed_texts([query])[0]
            
            # 유사도 검색
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # 결과 포맷팅
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'id': results['ids'][0][i]
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return []
    
    def get_collection_info(self, collection_name: str = "documents") -> Dict[str, Any]:
        """
        컬렉션 정보 조회
        
        Args:
            collection_name (str): 컬렉션 이름
            
        Returns:
            Dict: 컬렉션 정보
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            count = collection.count()
            return {
                'name': collection_name,
                'count': count,
                'exists': True
            }
        except Exception as e:
            logger.warning(f"컬렉션 정보 조회 실패: {e}")
            return {
                'name': collection_name,
                'count': 0,
                'exists': False
            }


def create_embedder(config_path: str = "config.yaml") -> DocumentEmbedder:
    """
    Document Embedder 인스턴스 생성
    
    Args:
        config_path (str): 설정 파일 경로
        
    Returns:
        DocumentEmbedder: Document Embedder 인스턴스
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return DocumentEmbedder(
        embedding_model_path=config['embedding_model_path'],
        chroma_db_path=config['chroma_db_path']
    )


if __name__ == "__main__":
    # 테스트 코드
    embedder = create_embedder()
    
    # 테스트 문서
    test_documents = [
        {
            'content': '인공지능(AI)은 인간의 학습능력과 추론능력, 지각능력을 인공적으로 구현한 컴퓨터 시스템입니다.',
            'metadata': {'source': 'test.txt', 'title': 'AI 정의'}
        },
        {
            'content': '머신러닝은 데이터로부터 패턴을 학습하여 예측이나 분류를 수행하는 기술입니다.',
            'metadata': {'source': 'test.txt', 'title': '머신러닝 정의'}
        }
    ]
    
    # 문서 추가
    embedder.add_documents(test_documents)
    
    # 검색 테스트
    results = embedder.search_similar("인공지능이란 무엇인가요?")
    
    print("검색 결과:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['document'][:100]}... (유사도: {1-result['distance']:.3f})") 