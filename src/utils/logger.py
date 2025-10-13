
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
from datetime import datetime
from typing import Optional

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

   logger.propagate = propagate
   logger.debug("Logger initialized (level=%s, log_dir=%s)", level, log_dir)
   return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
   """Return a logger by name (root logger if None)."""
   return logging.getLogger(name or "")
