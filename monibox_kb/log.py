import logging

def get_logger(name="monibox"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger