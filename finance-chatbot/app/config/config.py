import os
from dotenv import load_dotenv
from app.common.logger import get_logger

logger = get_logger(__name__)
logger.info("Loading Configuration")
# load environment variables
load_dotenv()

# assign model names
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_LLM_MODEL = "gpt-4o-mini"
HF_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# assign api keys
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINFACE_REPO_ID = os.environ.get("HUGGINFACE_REPO_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# assign data paths
DB_FAISS_PATH = "vector_db/faiss/"
DATA_PATH = "data/"


# set configuration parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
NUM_OF_DOCS_TO_RETRIEVE = 5
LLM_MODEL = "openai" #"hf", "openai"

logger.info("Configuration loaded successfully")
