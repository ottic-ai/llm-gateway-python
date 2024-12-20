import logging
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger('llm_gateway')
logger.setLevel(logging.DEBUG)

# Console handler with colored output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[34m',    # Blue
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.msg = f"{color}{record.msg}{self.COLORS['RESET']}"
        
        if hasattr(record, 'metadata') and record.metadata:
            record.msg = f"{record.msg} {record.metadata}"
            
        return super().format(record)

formatter = ColoredFormatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def log_error(message: str, error: Optional[Exception] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    if error:
        metadata = metadata or {}
        metadata.update({
            'error': str(error),
            'stack': getattr(error, '__traceback__', None)
        })
    logger.error(message, extra={'metadata': metadata})

def log_info(message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    logger.info(message, extra={'metadata': metadata})

def log_warn(message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    logger.warning(message, extra={'metadata': metadata})

def log_debug(message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    logger.debug(message, extra={'metadata': metadata})
