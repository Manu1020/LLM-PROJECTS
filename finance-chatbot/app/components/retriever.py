from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from app.components.llm_setup import initialize_llm
from app.components.vector_db import load_vector_db
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from cachetools import LRUCache

from app.config.config import NUM_OF_DOCS_TO_RETRIEVE

logger = get_logger(__name__)

SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant that can answer questions related to financial reports 
using only the information provided in the context. If you don't know the answer, 
just say that you don't know. Do not try to make up an answer.

Context:
{context}

Question: 
{question}

Answer:
"""
llm = initialize_llm()

# LRU cache for vector DBs
vector_db_cache = LRUCache(maxsize=5)

def get_vector_db_for_company(file_name=None):
    vector_db_name = "all" if file_name is None else file_name
    if vector_db_name not in vector_db_cache:
        vector_db_cache[vector_db_name] = load_vector_db(vector_db_name)
    return vector_db_cache[vector_db_name]

def get_prompt():
    return PromptTemplate(
        template=SYSTEM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

def build_qa_chain(file_name=None):
    try:
        vector_db = get_vector_db_for_company(file_name=file_name)
        if vector_db is None:
            raise CustomException("Vector database not found")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=vector_db.as_retriever(
                search_kwargs={"k": NUM_OF_DOCS_TO_RETRIEVE}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": get_prompt()}
        )
        logger.info("QA Retriever initialized successfully")
        return qa_chain
    except Exception as e:
        error_message = CustomException("Failed to initialize retriever", e)
        logger.error(str(error_message))
        raise error_message