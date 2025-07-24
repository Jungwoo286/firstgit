import os
from pathlib import Path

def to_abs_path(rel_path: str) -> str:
    """
    상대경로를 절대경로로 변환합니다.
    Args:
        rel_path (str): 상대경로
    Returns:
        str: 절대경로
    """
    return str(Path(rel_path).resolve())

def is_valid_path(path: str) -> bool:
    """
    경로가 실제로 존재하는지 확인합니다.
    Args:
        path (str): 확인할 경로
    Returns:
        bool: 존재 여부
    """
    return Path(path).exists() 