
"""
This module provides logging utilities for the bone age prediction project.

It includes functions to set up logging, mirror Keras output to files, and get loggers.
"""

import logging
import os
import sys
import threading
from datetime import datetime
from importlib import import_module
from typing import Optional, Sequence, Tuple

_FILE_HANDLER: Optional[logging.Handler] = None
_KERAS_STDOUT_PATCHED = False
_KERAS_PATCH_LOCK = threading.Lock()
_MODULE_LOGGER = logging.getLogger(__name__)

def setup_logging(
   log_dir: Optional[str] = None,
   level: str = "INFO",
   name: Optional[str] = None,
   propagate: bool = False,
) -> logging.Logger:
   """
   Set up logging with console and optional file output.

   Parameters
   ----------
   log_dir : Optional[str]
      Directory to save log files.
   level : str
      Logging level.
   name : Optional[str]
      Logger name.
   propagate : bool
      Whether to propagate logs to parent loggers.

   Returns
   -------
   logging.Logger
      The configured logger.
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
   logger.info("Logger initialized (level=%s, log_dir=%s)", level, log_dir)
   return logger


def _detect_file_handler() -> Optional[logging.Handler]:
   """
   Detect an existing file handler in the root logger.

   Returns
   -------
   Optional[logging.Handler]
      The file handler if found, else None.
   """
   for handler in logging.getLogger().handlers:
      if isinstance(handler, logging.FileHandler):
         return handler
   return None


def mirror_keras_stdout_to_file(level: str = "INFO") -> None:
   """
   Mirror Keras stdout output to the log file.

   Parameters
   ----------
   level : str
      Logging level for mirrored messages.
   """
   global _KERAS_STDOUT_PATCHED, _FILE_HANDLER

   with _KERAS_PATCH_LOCK:
      if _KERAS_STDOUT_PATCHED:
         return

      if _FILE_HANDLER is None:
         _FILE_HANDLER = _detect_file_handler()

      if _FILE_HANDLER is None:
         _MODULE_LOGGER.debug("No file handler detected; skipping Keras stdout mirroring.")
         return

      numeric_level = getattr(logging, level.upper(), logging.INFO)
      capture_logger = logging.getLogger("keras.callbacks.mirror")
      capture_logger.setLevel(numeric_level)
      capture_logger.propagate = False
      capture_logger.handlers.clear()
      capture_logger.addHandler(_FILE_HANDLER)

      module_names = (
         "keras.src.utils.io_utils",
         "keras.utils.io_utils",
         "tensorflow.keras.utils.io_utils",
         "tensorflow.python.keras.utils.io_utils",
      )

      targets: Sequence[Tuple[object, object]] = []
      failures: list[tuple[str, Exception]] = []
      for module_name in module_names:
         try:
            module = import_module(module_name)
         except Exception as exc:
            failures.append((module_name, exc))
            continue

         original = getattr(module, "print_msg", None)
         if original is None:
            continue
         targets.append((module, original))

      if not targets:
         if failures:
            details = ", ".join(f"{name}: {exc}" for name, exc in failures)
            _MODULE_LOGGER.warning(
               "Unable to import Keras io_utils modules (%s); skipping stdout mirroring.",
               details,
            )
         else:
            _MODULE_LOGGER.warning("Unable to import Keras io_utils; skipping stdout mirroring.")
         return

      primary_original = targets[0][1]

      def print_msg_with_mirror(message, line_break: bool = True, **kwargs):
         primary_original(message, line_break=line_break, **kwargs)
         if not line_break:
            return
         text = (message or "").strip()
         if not text:
            return
         for line in text.splitlines():
            clean_line = line.strip()
            if clean_line:
               capture_logger.log(numeric_level, clean_line)

      for module, _ in targets:
         setattr(module, "print_msg", print_msg_with_mirror)

      _KERAS_STDOUT_PATCHED = True
      _MODULE_LOGGER.info(
         "Mirroring Keras stdout to %s", getattr(_FILE_HANDLER, "baseFilename", "<memory>")
      )


def get_logger(name: Optional[str] = None) -> logging.Logger:
   """
   Get a logger instance.

   Parameters
   ----------
   name : Optional[str]
      Logger name.

   Returns
   -------
   logging.Logger
      The logger instance.
   """
   return logging.getLogger(name or "")
