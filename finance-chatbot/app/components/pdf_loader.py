import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)

def load_pdf_files(file_name=None):
    try:
        logger.info(f"File name: {file_name}")
        if file_name:
            data_path = os.path.join(DATA_PATH, f"{file_name}.pdf")
            glob_pattern = f"{file_name}.pdf"
        else:
            data_path = DATA_PATH
            glob_pattern = "*.pdf"

        if not os.path.exists(data_path):
            raise CustomException(f"Data path {data_path} does not exist")

        logger.info(f"Loading PDF files from {data_path}")
        loader = DirectoryLoader(DATA_PATH, glob=glob_pattern,loader_cls=PyPDFLoader)

        documents = loader.load()
        if not documents:
            logger.warning("No PDF files found in the data path {data_path}")
        else:
            logger.info(f"Successfully loaded {len(documents)} PDF files")

        return documents
    except Exception as e:
        error_message = CustomException("Error loading PDF files", e)
        logger.error(str(error_message))
        raise error_message

def create_text_chunks(documents):
    try:
        if not documents:
            raise CustomException("No documents provided")
        logger.info(f"Creating text chunks for {len(documents)} documents")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP)

        text_chunks = text_splitter.split_documents(documents)
        logger.info(f"Successfully created {len(text_chunks)} text chunks")
        return text_chunks
    except Exception as e:
        error_message = CustomException("Error creating text chunks", e)
        logger.error(str(error_message))
        raise error_message
