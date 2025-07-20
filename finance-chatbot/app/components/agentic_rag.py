from app.components.retriever import build_qa_chain
from app.components.tavily_search import tavily_lookup
from app.common.logger import get_logger
from app.components.llm_setup import initialize_llm

logger = get_logger(__name__)
llm = initialize_llm()
from pydantic import BaseModel, Field


class IsDefinitionResponse(BaseModel):
    is_definition_query: bool = Field(..., description="Is the query a request for a definition of a financial term or concept?")

def is_definition_query(user_query):
    prompt = (
        f"User query: {user_query}\n"
        "Is this a request for a definition of a financial term or concept? "
        "Answer True or False."
    )
    response = llm.with_structured_output(IsDefinitionResponse).invoke(prompt)
    logger.info(f"Is definition query: {response.is_definition_query}")
    return response.is_definition_query

def rewrite_query(query, history):
    try:
        prompt = (
            f"Conversation so far:\n{history}\n"
            f"Original Question: {query}\n"
            "Rewrite the original question as a clear, concise, natural-language query suitable for searching a financial report.\n"
            "\n"
            "### Guidelines:\n"
            "- Use standard financial definitions for all terms (e.g., use 'net income' for 'earnings', 'revenue' for 'sales', etc.).\n"
            "- Only use information from the conversation history if the original question refers to previous discussion. \n"
            "- If the original question is clear and self-contained, ignore the conversation history and rewrite based only on the current question.\n"
            "- Do NOT use or substitute values or components from previous questions or answers unless the user explicitly refers to them.\n"
            "- Only include financial components that are explicitly mentioned or required by the original question.\n"
            "- If the question requires a calculation (e.g., a financial ratio), rewrite it to explicitly request the necessary input values and the final result, using standard definitions.\n"
            "- Briefly describe the calculation logic in plain English (e.g., 'net income divided by number of shares'), but do NOT provide step-by-step instructions or formulas.\n"
            "- Do NOT ask the user to provide any values or make assumptions.\n"
            "- Refer only to actual company data as reported—no examples or hypotheticals.\n"
            "- Format the rewritten query as a single, well-formed question that could be answered using a financial document.\n"
        )

        response = llm.invoke(prompt)
        
        # Validate response
        if not response or not hasattr(response, 'content'):
            logger.error("Invalid response from llm in rewrite_query")
            return None
            
        content = response.content
        if not isinstance(content, str):
            content = str(content)
            logger.warning(f"Content was not a string, converted to: {content}")
        
        logger.info(f"Rewritten query: {content}")
        return content
        
    except Exception as e:
        logger.error(f"Error rewriting query: {e}")
        return None

def format_history(history):
    if not history:
        return "Beginning of the conversation"
    return "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in history
    )

def agentic_rag_pipeline(user_query, file_name, history=None):
    try:
        formatted_history = format_history(history)
        retrieval_query = rewrite_query(user_query, formatted_history)
        qa_chain = build_qa_chain(file_name=file_name)
        formatted_query = (
            "### Rewritten query based on history:\n"
            f"{retrieval_query}\n"
            f"Original Question: {user_query}"
        )
        logger.info(f"Formatted Query: {formatted_query}")
        response = qa_chain.invoke({
            "query": formatted_query
        })
        if not response or not isinstance(response, dict):
            logger.error("Invalid response from qa_chain")
            return {"response": "Sorry, I encountered an error processing your request.", "sources": []}
        result = response.get("result", "No response")
        if not isinstance(result, str):
            result = str(result)
            logger.warning(f"Result was not a string, converted to: {result}")
        return {"response": result, "sources": response.get("source_documents", [])}
        
    except Exception as e:
        logger.error(f"Error in agentic_rag_pipeline: {e}")
        return {"response": "Sorry, I encountered an error processing your request.", "sources": []}


if __name__ == "__main__":
    from app.components.create_index import create_index
    file_name="NASDAQ_AAPL_2024"
    create_index(file_name=file_name)
    result = agentic_rag_pipeline("What is percentage increase or decrease in operating income?", file_name="NASDAQ_AAPL_2024")
    print(result["response"])