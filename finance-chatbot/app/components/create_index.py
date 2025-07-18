import os

from app.components.pdf_loader import load_pdf_files, create_text_chunks
from app.components.vector_db import create_vector_db
from app.config.config import DB_FAISS_PATH

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def create_index(file_name=None, force_reindex=False):
    """
    Creates a vector index for the given file or all files.
    If the index already exists and force_reindex is False, skips creation.
    """
    try:
        # Determine the index path
        index_name = file_name if file_name else "all"
        index_path = os.path.join(DB_FAISS_PATH, index_name)
        logger.info(f"Index path: {index_path}")
        # Check if index already exists
        if os.path.exists(index_path) and not force_reindex:
            logger.info(f"Index already exists at {index_path}. Skipping re-indexing.")
            return
        
        logger.info("Loading data")
        documents = load_pdf_files(file_name)
        text_chunks = create_text_chunks(documents)
        create_vector_db(text_chunks, file_name)
        logger.info("Index created successfully")
    except Exception as e:
        error_message = CustomException("Failed to create index", e)
        logger.error(str(error_message))
        raise error_message

if __name__ == "__main__":
    file_name = "NASDAQ_AMZN_2024"
    create_index(file_name)