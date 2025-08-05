# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib.util
import logging
import os
from contextlib import contextmanager
from types import MethodType
from typing import Optional

from modelscope.utils.logger import get_logger as get_ms_logger


# Avoid circular reference
def _is_local_master():
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    return local_rank in {-1, 0}


init_loggers = {}

# old format
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_format = logging.Formatter('[%(levelname)s:%(name)s] %(message)s')

info_set = set()
warning_set = set()


def info_if(self, msg, cond, *args, **kwargs):
    if cond:
        with logger_context(self, logging.INFO):
            self.info(msg)


def warning_if(self, msg, cond, *args, **kwargs):
    if cond:
        with logger_context(self, logging.INFO):
            self.warning(msg)


def info_once(self, msg, *args, **kwargs):
    hash_id = kwargs.get('hash_id') or msg
    if hash_id in info_set:
        return
    info_set.add(hash_id)
    self.info(msg)


def warning_once(self, msg, *args, **kwargs):
    hash_id = kwargs.get('hash_id') or msg
    if hash_id in warning_set:
        return
    warning_set.add(hash_id)
    self.warning(msg)


def get_logger(log_file: Optional[str] = None, log_level: Optional[int] = None, file_mode: str = 'w'):
    import colorlog
    """Get logging logger with colored output

    Args:
        log_file: Log pathname, if specified, file handler will be added to logger
        log_level: Logging level.
        file_mode: Specifies the mode to open the file
    """
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'DEBUG').upper()  # 默认DEBUG级别
        log_level = getattr(logging, log_level, logging.DEBUG)  # 允许控制台控制级别
    
    logger_name = __name__.split('.')[0]
    logger = logging.getLogger(logger_name)
    logger.propagate = False

    # 清除现有的handlers，避免重复
    if logger.handlers:
        logger.handlers.clear()

    # 创建彩色控制台处理器
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)  # 显式允许 DEBUG 级别

    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(pathname)s:%(lineno)d] %(message)s%(reset)s",  
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        reset=True,
        style='%'
    )
    stream_handler.setFormatter(color_formatter)
    logger.addHandler(stream_handler)

    # 如果需要文件日志
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        file_handler.setLevel(logging.DEBUG)  # 允许所有日志级别写入文件
        file_handler.setFormatter(logging.Formatter('[%(levelname)s:%(pathname)s:%(lineno)d] %(message)s'))
        logger.addHandler(file_handler)

    # 确保 logger 级别不会限制 DEBUG
    logger.setLevel(logging.DEBUG)

    # 添加一次性打印方法
    def info_once(self, msg, *args, **kwargs):
        if not hasattr(self, "_info_logged"):
            self._info_logged = set()
        if msg not in self._info_logged:
            self._info_logged.add(msg)
            self.info(msg, *args, **kwargs)

    def warning_once(self, msg, *args, **kwargs):
        if not hasattr(self, "_warning_logged"):
            self._warning_logged = set()
        if msg not in self._warning_logged:
            self._warning_logged.add(msg)
            self.warning(msg, *args, **kwargs)

    logger.info_once = MethodType(info_once, logger)
    logger.warning_once = MethodType(warning_once, logger)
    logger.info_if = MethodType(info_if, logger)
    logger.warning_if = MethodType(warning_if, logger)
    return logger


logger = get_logger()
ms_logger = get_ms_logger()

logger.handlers[0].setFormatter(logger_format)
ms_logger.handlers[0].setFormatter(logger_format)
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
if _is_local_master():
    ms_logger.setLevel(log_level)
else:
    ms_logger.setLevel(logging.ERROR)


@contextmanager
def logger_context(logger, log_leval):
    origin_log_level = logger.level
    logger.setLevel(log_leval)
    try:
        yield
    finally:
        logger.setLevel(origin_log_level)


@contextmanager
def ms_logger_context(log_leval):
    with logger_context(get_ms_logger(), log_leval):
        yield


def add_file_handler_if_needed(logger, log_file, file_mode, log_level):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return

    if importlib.util.find_spec('torch') is not None:
        is_worker0 = int(os.getenv('LOCAL_RANK', -1)) in {-1, 0}
    else:
        is_worker0 = True

    if is_worker0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        file_handler.setFormatter(logger_format)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
