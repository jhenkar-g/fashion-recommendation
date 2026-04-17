"""Logging configuration"""
import logging
from pathlib import Path

def setup_logging(log_dir='logs', log_file='training.log'):
    Path(log_dir).mkdir(exist_ok=True)
    log_path = Path(log_dir) / log_file
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    return logging.getLogger(__name__)

logger = setup_logging()
