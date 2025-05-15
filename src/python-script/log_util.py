# log_util.py

import logging
import os
import sys

def setup_local_logger(log_path="logs/run.log", logger_name="default"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # Prevent adding handlers multiple times
        formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler (optional)
        if logger_name == "default":
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Redirect print only once for main logger
        if logger_name == "default":
            import builtins
            builtins.print = lambda *args, **kwargs: logger.info(" ".join(str(a) for a in args))

    return logger
