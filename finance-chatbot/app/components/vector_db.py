import os
from langchain_community.vectorstores import FAISS

from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)
embedding_model = get_embedding_model()
if embedding_model is None:
    raise CustomException("Embedding model not found")

def load_vector_db(file_name=None):
    try:
        if file_name:
            db_path = os.path.join(DB_FAISS_PATH, file_name)
        else:
            db_path = os.path.join(DB_FAISS_PATH, 'all')

        if os.path.exists(db_path):
            logger.info(f"Loading vector database from {db_path}")
            return FAISS.load_local(
                db_path,
                embedding_model,
                allow_dangerous_deserialization=True)
        else:
            logger.info(f"No vector database found at {DB_FAISS_PATH}")
            return None
    except Exception as e:
        error_message = CustomException("Error loading vector database", e)
        logger.error(str(error_message))
        return None


def create_vector_db(text_chunks, file_name=None):
    try:
        if not text_chunks:
            raise CustomException("No text chunks provided")
        logger.info(f"Creating vector database with {len(text_chunks)} text chunks")
        if file_name:
            db_path = os.path.join(DB_FAISS_PATH, file_name)
        else:
            db_path = os.path.join(DB_FAISS_PATH, 'all')
        db = FAISS.from_documents(text_chunks, embedding_model)
        db.save_local(db_path)       
        logger.info(f"Successfully created vector database at {db_path}")
        return db
    except Exception as e:
        error_message = CustomException("Error creating vector database", e)
        logger.error(str(error_message))
        return None
        
