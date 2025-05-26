# log_util.py

import logging
import os
import sys
import builtins


def setup_local_logger(log_path="logs/run.log", logger_name="default"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # Prevent adding handlers multiple times
        formatter = logging.Formatter(
            "[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Optional: print to console
        if logger_name == "default":
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # ✅ Safe print override to prevent infinite recursion
            original_print = builtins.print

            def safe_print(*args, **kwargs):
                msg = " ".join(str(a) for a in args)
                try:
                    logger.info(msg)
                except Exception:
                    original_print(msg)
            builtins.print = safe_print

    return logger
