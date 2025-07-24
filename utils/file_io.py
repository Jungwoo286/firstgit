import os
from pathlib import Path
from datetime import datetime

def ensure_dir(path: str) -> None:
    """
    주어진 경로의 폴더가 없으면 생성합니다.
    Args:
        path (str): 생성할 폴더 경로
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"[안내] 폴더 자동생성: {path}")
    except Exception as e:
        log_error(f"폴더 생성 실패: {path} - {e}")

def log_error(msg: str) -> None:
    """
    에러 메시지를 logs/error.log에 기록합니다.
    Args:
        msg (str): 에러 메시지
    """
    try:
        ensure_dir('logs')
        with open('logs/error.log', 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now()}] {msg}\n")
    except Exception as e:
        print(f"[Error] 로그 기록 실패: {e}")

def read_file(filepath: str, encoding: str = 'utf-8') -> str:
    """
    파일을 읽어 문자열로 반환합니다.
    Args:
        filepath (str): 파일 경로
        encoding (str): 인코딩 방식
    Returns:
        str: 파일 내용
    Raises:
        FileNotFoundError, UnicodeDecodeError
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        log_error(f"파일 읽기 실패: {filepath} - {e}")
        return ''

def write_file(filepath: str, content: str, encoding: str = 'utf-8') -> None:
    """
    문자열을 파일로 저장합니다.
    Args:
        filepath (str): 저장할 파일 경로
        content (str): 저장할 내용
        encoding (str): 인코딩 방식
    """
    try:
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)
    except Exception as e:
        log_error(f"파일 저장 실패: {filepath} - {e}") 