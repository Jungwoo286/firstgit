"""
LLM Runner Module
EEVE Korean Instruct 2.8B 모델을 사용하여 답변을 생성하는 모듈
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMRunner:
    """
    EEVE Korean Instruct 2.8B 모델을 사용하여 답변을 생성하는 클래스
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        LLM Runner 초기화
        
        Args:
            model_path (str): 모델 경로
            device (str, optional): 사용할 디바이스 ('cuda', 'cpu')
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"모델 로딩 중: {model_path}")
        logger.info(f"사용 디바이스: {self.device}")
        
        try:
            # 토크나이저 로딩
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # 모델 로딩
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map='auto' if self.device == 'cuda' else None,
                trust_remote_code=True
            )
            
            if self.device == 'cpu':
                self.model = self.model.to(self.device)
            
            logger.info("모델 로딩 완료")
            
        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            raise
    
    def generate_answer(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """
        프롬프트를 기반으로 답변 생성
        
        Args:
            prompt (str): 입력 프롬프트
            max_length (int): 최대 토큰 길이
            temperature (float): 생성 온도 (0.0 ~ 1.0)
            
        Returns:
            str: 생성된 답변
        """
        try:
            # 입력 토크나이징
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048  # 입력 최대 길이
            )
            
            if self.device == 'cpu':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 답변 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # 토큰을 텍스트로 디코딩
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            return "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다."
    
    def generate_rag_answer(self, question: str, context: List[str]) -> str:
        """
        RAG 시스템을 위한 답변 생성
        
        Args:
            question (str): 사용자 질문
            context (List[str]): 검색된 관련 문단들
            
        Returns:
            str: 생성된 답변
        """
        # 컨텍스트 결합
        context_text = "\n\n".join(context)
        
        # EEVE 모델용 프롬프트 템플릿
        prompt = f"""<|im_start|>system
당신은 도움이 되는 AI 어시스턴트입니다. 주어진 문서를 기반으로 정확하고 유용한 답변을 제공하세요.
<|im_end|>
<|im_start|>user
다음 문서를 참고하여 질문에 답변해주세요:

문서:
{context_text}

질문: {question}
<|im_end|>
<|im_start|>assistant
"""
        
        return self.generate_answer(prompt)
    
    def __del__(self):
        """소멸자: 메모리 정리"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def create_llm_runner(model_path: str) -> LLMRunner:
    """
    LLM Runner 인스턴스 생성
    
    Args:
        model_path (str): 모델 경로
        
    Returns:
        LLMRunner: LLM Runner 인스턴스
    """
    return LLMRunner(model_path)


if __name__ == "__main__":
    # 테스트 코드
    import yaml
    
    # 설정 파일 로딩
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # LLM Runner 생성
    llm = create_llm_runner(config['llm_model_path'])
    
    # 테스트 질문
    test_question = "안녕하세요! 간단한 인사말을 해주세요."
    test_context = ["이 문서는 AI 어시스턴트에 대한 설명입니다."]
    
    # 답변 생성
    answer = llm.generate_rag_answer(test_question, test_context)
    print(f"질문: {test_question}")
    print(f"답변: {answer}") 