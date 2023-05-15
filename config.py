import os
from distutils.util import strtobool

from utils import Singleton
from dotenv import load_dotenv

load_dotenv()

class Config(metaclass=Singleton):
    def __init__(self) -> None:
         self.openai_api_key = os.getenv("OPENAI_API_KEY")
         self.model_name = os.getenv("MODEL_NAME")
         self.detection_model = os.getenv("DETECTION_MODEL")
         self.segmentation_model = os.getenv("SEGMENTATION_MODEL")
         self.download_models = os.getenv("DOWNLOAD_MODELS")
         self.models_path = os.getenv("MODELS_PATH")
         self.webcam_index = int(os.getenv("WEBCAM_INDEX"))
         self.log_path = os.getenv("LOG_PATH")
         self.log_file = os.getenv("LOG_FILE")
         self.output_file = os.getenv("OUTPUT_FILE")
         self.output_embeddings = os.getenv("OUTPUT_EMBEDDINGS")
         self.log_file_clean_at_launch = bool(strtobool(os.getenv("LOG_FILE_CLEAN_AT_LAUNCH")))  # noqa: E501

