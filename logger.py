import logging
import os

from utils import Singleton
from config import Config

CFG = Config()

class Logger(metaclass=Singleton):
    def __init__(self, name: str, level: int = logging.INFO):
        if not os.path.exists(CFG.log_path):
            os.makedirs(CFG.log_path)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # noqa: E501

        if CFG.log_file:
            if CFG.log_file_clean_at_launch:
                if os.path.exists(CFG.log_path + CFG.log_file):
                    os.remove(CFG.log_path + CFG.log_file)
            
            # Create the log file if it doesn't exist
            file_handler = logging.FileHandler(CFG.log_path + CFG.log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        else:
            # Log to console only
            self.logger.warning("No log file specified. Logging to console only!")
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
        
    def get_logger(self) -> logging.Logger:
        return self.logger

logger = Logger('yologpt').get_logger()