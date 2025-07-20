from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from app.components.llm_setup import initialize_llm
from app.components.vector_db import load_vector_db
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from cachetools import LRUCache

from app.config.config import NUM_OF_DOCS_TO_RETRIEVE

logger = get_logger(__name__)

PROMPT_TEMPLATE = """
You are a precise and grounded assistant that answers financial report questions using only the context provided.

## Instructions

1. **Choose the Right Query**
   - You are given both the original and a rewritten version of the user’s question.
   - **Use the original question if it is clear and directly answerable from the context.**
   - Use the rewritten version **only if** the original is ambiguous or incomplete.
   - Do not perform calculations if a direct value can be extracted from the context.
   - Do not mention or explain which version of the question you are using, and do not include any commentary about your reasoning or process.


2. **Extract Information**
   - Identify and extract only the values explicitly mentioned in the query.
   - Preserve original formatting and units (e.g., "$3.2M", "18%").

3. **If Calculation is Required**
   - Only perform a calculation if ALL required values are present AND the value is not already provided directly.
   - Keep the explanation minimal—just describe the logic in plain English (no formulas).
   - Do simple arithmetic and provide the final result with appropriate units.

4. **If Any Value is Missing**
   - Clearly state which value(s) are missing.
   - Do not guess, fabricate, or request user input.

5. **Stay Grounded**
   - Rely strictly on the given context.
   - Do not use external knowledge, assumptions, or commentary.

---

### Context:
{context}

{question}

---

## Final Answer:
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
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

def build_qa_chain(file_name=None):
    try:
        vector_db = get_vector_db_for_company(file_name=file_name)
        if vector_db is None:
            raise CustomException("Vector database not found")
        
        retriever = vector_db.as_retriever(search_kwargs={"k": NUM_OF_DOCS_TO_RETRIEVE})


        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": get_prompt()}
        )
        logger.info("QA Retriever initialized successfully")
        return qa_chain
    except Exception as e:
        error_message = CustomException("Failed to initialize retriever", e)
        logger.error(str(error_message))
        raise error_message