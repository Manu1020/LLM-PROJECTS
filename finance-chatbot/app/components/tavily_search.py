from langchain_tavily import TavilySearch
from app.config.config import TAVILY_API_KEY
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def tavily_lookup(query):
    try:
        tool = TavilySearch(
            api_key=TAVILY_API_KEY,
            max_results=1,
            topics=["finance"],
            include_domains=["investopedia.com"]
        )
        response = tool.run(query)
        logger.info(f"Tavily results: {response["results"][0]["content"]}")
        return response["results"][0]["content"]
    except Exception as e:
        error_message = CustomException("Error calling Tavily API", e)
        logger.error(str(error_message))
        return None

if __name__ == "__main__":
    print(tavily_lookup("What is the P/E ratio?"))