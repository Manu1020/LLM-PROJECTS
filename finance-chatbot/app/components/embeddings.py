from langchain_huggingface import HuggingFaceEmbeddings

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import HF_EMBEDDING_MODEL

logger = get_logger(__name__)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def get_embedding_model():
    try:
        logger.info("Loading embedding model")
        embedding_model = HuggingFaceEmbeddings(
            model_name=HF_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        logger.info("Successfully loaded embedding model")
        return embedding_model
    except Exception as e:
        error_message = CustomException("Error loading embedding model", e)
        logger.error(str(error_message))
        return None