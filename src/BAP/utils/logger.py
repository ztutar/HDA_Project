
"""Utility helpers for consistent, application-wide logging configuration.

This module centralizes logging setup for both console and file outputs and
provides a lightweight bridge that mirrors verbose Keras callback stdout
messages back into the active log file.  The helpers are designed to be
imported anywhere in the codebase so that loggers remain uniform (same
formatting, timestamps, levels) while still allowing each caller to choose
its own logger name.

Functions exposed here are intentionally side-effectful: `setup_logging`
configures handlers on the root logger once, `mirror_keras_stdout_to_file`
monkey-patches Keras utilities exactly once in a thread-safe fashion, and
`get_logger` simply returns a logger by name without adding handlers.  
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
   """Configure and return a logger ready for console/file output.

   Parameters
   ----------
   log_dir:
      Optional directory where a timestamped log file should be created. The
      directory is created if it does not exist. When omitted, only console
      logging is enabled.
   level:
      Logging level string understood by the `logging` module (e.g. ``"INFO"``,
      ``"DEBUG"``). Anything unknown defaults to ``logging.INFO``.
   name:
      Name of the logger to configure. Passing ``None`` or an empty string
      targets the root logger so that child loggers inherit the handlers.
   propagate:
      Whether messages emitted by the configured logger should bubble up to
      parent loggers. Defaults to ``False`` to avoid duplicate entries.

   Returns
   -------
   logging.Logger
      The configured logger instance. Subsequent calls with the same name will
      return the existing logger without re-adding handlers.
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
   """Return the first file handler attached to the root logger, if any.

   This helper is used when Keras stdout mirroring needs to piggyback on the
   already-configured file handler instead of creating a new one.  It keeps
   the mirroring logic lightweight and avoids leaking duplicate files.
   """
   for handler in logging.getLogger().handlers:
      if isinstance(handler, logging.FileHandler):
         return handler
   return None


def mirror_keras_stdout_to_file(level: str = "INFO") -> None:
   """Mirror Keras callback stdout messages into the active log file.

   Keras callback internals frequently write progress updates via their own
   `io_utils.print_msg` helper, bypassing Python's logging framework.  This
   function monkey-patches every discoverable variant of that helper so that
   emitted lines also flow into the same file handler configured by
   `setup_logging`.  The patch is applied once in a thread-safe manner and is
   skipped entirely when no file handler is available.

   Parameters
   ----------
   level:
      Logging level used when replaying the mirrored stdout lines. The default
      ``"INFO"`` level keeps parity with most Keras progress messages.
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
   """Return an existing logger without modifying handlers.

   Parameters
   ----------
   name:
      Logger name to retrieve. When ``None`` (or empty), the root logger is
      returned, allowing modules to reuse the configuration produced by
      `setup_logging`.

   Returns
   -------
   logging.Logger
      The existing logger instance; no handlers or levels are altered.
   """
   return logging.getLogger(name or "")
