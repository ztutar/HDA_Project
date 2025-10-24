
"""
This module centralizes logging setup for the application and exposes helpers that
keep log configuration consistent. It builds a logger with both console and optional
file handlers, using readable timestamps and log level metadata. The module offers 
two main entry points:

-  `setup_logging` prepares the logging environment, creates the handlers, and ensures
   all emitted messages respect the chosen verbosity and formatting.
-  `get_logger` returns a named logger (or the root logger) so other modules can reuse
   the shared configuration without duplicating setup code.
"""

import logging
import os
import sys
import threading
from datetime import datetime
from typing import Optional


_FILE_HANDLER: Optional[logging.Handler] = None
_KERAS_STDOUT_PATCHED = False
_KERAS_PATCH_LOCK = threading.Lock()

def setup_logging(
   log_dir: Optional[str] = None,
   level: str = "INFO",
   name: Optional[str] = None,
   propagate: bool = False,
) -> logging.Logger:
   """
   Configure application logging with console and optional file handler.
   If the target logger already has handlers, it will be reused.
   Args:
      log_dir: Directory where a timestamped log file is created. If None, file logging is disabled.
      level: Log level name (e.g., "DEBUG", "INFO").
      name: Logger name. Use None for root logger.
      propagate: Whether logs propagate to ancestor loggers.
   Returns:
      logging.Logger: Configured logger instance.
   """
   global _FILE_HANDLER

   logger = logging.getLogger(name or "")
   if logger.handlers:
      return logger

   numeric_level = getattr(logging, level.upper(), logging.INFO)
   logger.setLevel(numeric_level)

   fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
   datefmt = "%Y-%m-%d %H:%M:%S"
   formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

   console_handler = logging.StreamHandler(stream=sys.stdout)
   console_handler.setLevel(numeric_level)
   console_handler.setFormatter(formatter)
   logger.addHandler(console_handler)

   if log_dir:
      os.makedirs(log_dir, exist_ok=True)
      ts = datetime.now().strftime("%Y%m%d_%H%M%S")
      file_path = os.path.join(log_dir, f"run_{ts}.log")
      file_handler = logging.FileHandler(file_path, encoding="utf-8")
      file_handler.setLevel(numeric_level)
      file_handler.setFormatter(formatter)
      logger.addHandler(file_handler)
      _FILE_HANDLER = file_handler

   logger.propagate = propagate
   logger.debug("Logger initialized (level=%s, log_dir=%s)", level, log_dir)
   return logger


def _detect_file_handler() -> Optional[logging.Handler]:
   """Return the first file handler attached to the root logger, if any."""
   for handler in logging.getLogger().handlers:
      if isinstance(handler, logging.FileHandler):
         return handler
   return None


def mirror_keras_stdout_to_file(level: str = "INFO") -> None:
   """Mirror Keras callback stdout messages into the active log file."""
   global _KERAS_STDOUT_PATCHED, _FILE_HANDLER

   with _KERAS_PATCH_LOCK:
      if _KERAS_STDOUT_PATCHED:
         return

      if _FILE_HANDLER is None:
         _FILE_HANDLER = _detect_file_handler()

      if _FILE_HANDLER is None:
         return

      try:
         from tensorflow.keras.utils import io_utils  # type: ignore import
      except Exception:
         return

      numeric_level = getattr(logging, level.upper(), logging.INFO)
      capture_logger = logging.getLogger("keras.callbacks.mirror")
      capture_logger.setLevel(numeric_level)
      capture_logger.propagate = False
      capture_logger.handlers.clear()
      capture_logger.addHandler(_FILE_HANDLER)

      original_print_msg = io_utils.print_msg

      def print_msg_with_mirror(message, line_break: bool = True, **kwargs):
         original_print_msg(message, line_break=line_break, **kwargs)
         if not line_break:
            return
         text = (message or "").strip()
         if not text:
            return
         for line in text.splitlines():
            clean_line = line.strip()
            if clean_line:
               capture_logger.log(numeric_level, clean_line)

      io_utils.print_msg = print_msg_with_mirror
      _KERAS_STDOUT_PATCHED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
   """Return a logger by name (root logger if None)."""
   return logging.getLogger(name or "")
