"""
프롬프트 템플릿 관리 모듈
- system prompt, 사용자 질문, context 조합 템플릿
"""

def get_system_prompt() -> str:
    """
    시스템 프롬프트 템플릿 반환
    Returns:
        str: 시스템 프롬프트
    """
    return "You are a helpful assistant."

def build_structured_prompt(system_prompt: str, question: str, contexts: list) -> str:
    """
    시스템 프롬프트, 질문, context를 조합해 최종 프롬프트 생성
    Args:
        system_prompt (str): 시스템 프롬프트
        question (str): 사용자 질문
        contexts (list): 문단 리스트
    Returns:
        str: 완성된 프롬프트
    """
    # 예시 템플릿 조합 (구현 예정)
    pass 