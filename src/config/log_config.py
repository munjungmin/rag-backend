from contextvars import ContextVar
import os
from pathlib import Path
import socket

import structlog
import logging
import sys
from typing import Optional
from structlog.types import WrappedLogger, EventDict


# Context variables
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
project_id_var: ContextVar[Optional[str]] = ContextVar("project_id", default=None)

POD_NAME = os.getenv("POD_NAME", "local")  # 어느 서버 컨테이너인지 
HOST_NAME = socket.gethostname()           # 서버 호스트 이름 


# Helper Function 
def get_log_level() -> int:
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    return {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }.get(log_level_str, logging.INFO)


def add_context_info(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    request_id = request_id_var.get()
    if request_id:
        event_dict["request_id"] = request_id
    user_id = user_id_var.get()
    if user_id:
        event_dict["user_id"] = user_id
    project_id = project_id_var.get()
    if project_id:
        event_dict["project_id"] = project_id
    event_dict["pod_name"] = POD_NAME
    event_dict["host_name"] = HOST_NAME
    return event_dict


def rename_event_to_message(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    if "event" in event_dict:
        event_dict["message"] = event_dict.pop("event")
    return event_dict


def order_keys(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    key_order = ["timestamp", "level", "logger", "message", "func_name", "lineno"]
    ordered = {k: event_dict.pop(k) for k in key_order if k in event_dict}
    ordered.update(event_dict)
    return ordered

def configure_std_out_handler(root_logger) -> logging.Handler:
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(stdout_handler)

def configure_file_handler(root_logger, log_filename: str) -> logging.Handler:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / log_filename, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(file_handler)


# 핵심 함수 
def configure_logging(log_filename: str = "application.log") -> None:
    log_level = get_log_level() # priority 

    # 루트 로거 세팅 
    root_logger = logging.getLogger() # 최상위 로거, 모든 로거는 루트 로거를 상속받아 여기 설정하면 전체에 적용
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()     # 핸들러: 로그 출력 위치를 결정 

    # 핸들러 2개 등록 (터미널 출력 + 파일 저장)
    configure_std_out_handler(root_logger)   
    configure_file_handler(root_logger, log_filename)

    # 다른 라이브러리에서 출력되는 로그 레벨 조정 
    logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
    logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("celery").setLevel(logging.INFO)


    # 2) structlog: 앱 로그용
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level, # 설정된 로그 레벨보다 낮으면 skip
            structlog.contextvars.merge_contextvars, # 컨텍스트 변수를 로그에 자동 포함 
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False, key="timestamp"),
            structlog.processors.CallsiteParameterAdder(
                [
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO
                ]
            ),
            structlog.stdlib.add_log_level,  # 로그 레벨 출력 
            structlog.stdlib.add_logger_name, # 로거 이름 출력 
            add_context_info,
            rename_event_to_message,
            order_keys,
            structlog.processors.StackInfoRenderer(), # 예외 발생 시 스택 트레이스를 로그에 포함
            structlog.processors.format_exc_info,     # exc_info = True일때 예외 정보 포맷팅
            structlog.processors.JSONRenderer(ensure_ascii=False)  # dict -> JSON 문자열로 변환
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,  # Use stdlib logger wrapper
        cache_logger_on_first_use=True, #singleton optimization
    )

def get_logger(name: Optional[str] = None):
    return structlog.get_logger(name)


def set_request_id(request_id: str) -> None:
    request_id_var.set(request_id)

def set_user_id(user_id: str) -> None:
    user_id_var.set(user_id)

def set_project_id(project_id: str) -> None:
    project_id_var.set(project_id)

def clear_context() -> None:
    request_id_var.set(None)
    user_id_var.set(None)
    project_id_var.set(None)