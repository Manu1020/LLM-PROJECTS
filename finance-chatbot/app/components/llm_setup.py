from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import OPENAI_API_KEY, OPENAI_LLM_MODEL
from app.config.config import HF_TOKEN, HF_LLM_MODEL
from app.config.config import LLM_MODEL
import os

logger = get_logger(__name__)

def initialize_openai_llm():
    """Setup LLM components - renamed to match app.py expectations"""
    try:
        logger.info(f"Initializing OpenAI LLM: {OPENAI_LLM_MODEL}")
        llm = ChatOpenAI(
            model=OPENAI_LLM_MODEL,
            api_key=OPENAI_API_KEY
        )
        logger.info(f"OpenAI LLM initialized successfully")
        return llm
    except Exception as e:
        error_message = CustomException("Failed to initialize OpenAI LLMs", e)
        logger.error(str(error_message))
        raise error_message

def initialize_hf_llm():
    try:
        logger.info(f"Initializing HF LLM: {HF_LLM_MODEL}")
        llm = HuggingFaceEndpoint(
            repo_id=HF_LLM_MODEL,
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0, 
            max_new_tokens=512,
            return_full_text=False,
        )
        logger.info(f"HF LLM initialized successfully")
        return llm
    except Exception as e:  
        error_message = CustomException("Failed to initialize HF LLMs", e)
        logger.error(str(error_message))
        raise error_message

def initialize_llm():
    """Setup LLM components - renamed to match app.py expectations"""
    try:
        if LLM_MODEL == "openai":
            llm = initialize_openai_llm()
        elif LLM_MODEL == "hf":
            llm = initialize_hf_llm()
        else:
            raise CustomException(f"Invalid LLM model: {LLM_MODEL}")
        if llm is None:
            raise CustomException(f"Failed to initialize LLM")
        return llm
    except Exception as e:
        error_message = CustomException("Failed to initialize LLMs", e)
        logger.error(str(error_message))
        raise error_message