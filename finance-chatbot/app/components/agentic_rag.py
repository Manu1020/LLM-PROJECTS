from app.components.retriever import build_qa_chain
from app.components.tavily_search import tavily_lookup
from app.common.logger import get_logger
from app.components.llm_setup import initialize_llm

logger = get_logger(__name__)
llm = initialize_llm()
from pydantic import BaseModel, Field

class IsContextSufficientResponse(BaseModel):
    is_sufficient: bool = Field(..., description="Whether the context is sufficient to answer the question")

def is_context_sufficient(query, context):
    prompt = (
        f"Question: {query}\n"
        f"Context: {context}\n"
        "Is this context sufficient to answer the question? Return True or False"
    )
    response = llm.with_structured_output(IsContextSufficientResponse).invoke(prompt)
    logger.info(f"Is context sufficient: {response.is_sufficient}")
    return response.is_sufficient

def rewrite_query(query, definition):
    try:
        prompt = (
            f"Original Question: {query}\n"
            f"Definition: {definition}\n\n"
            "Rewrite the original question as a single, clear, natural-language query suitable for searching a financial report. "
            "The rewritten query should do the following:\n"
            "- List all the specific financial parameters or components needed to answer the question, based on the definition.\n"
            "- Clearly state, in plain English, how to calculate the answer from those parameters (i.e., provide the calculation formula in words, not symbols or LaTeX).\n"
            "- Do NOT use any example or hypothetical values or numbers from the definition.\n"
            "- Do NOT ask the user to provide any values or information.\n"
            "- The query should refer only to the actual company or report, not hypothetical or sample values.\n"
            "Format the rewritten query as a single, well-formed question that could be answered by a financial document."
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

def agentic_rag_pipeline(user_query, file_name=None):
    try:
        # Step 1: Try to answer from vector DB
        qa_chain = build_qa_chain(file_name=file_name)
        response = qa_chain.invoke({"query": user_query})
        
        # Validate response
        if not response or not isinstance(response, dict):
            logger.error("Invalid response from qa_chain")
            return {"response": "Sorry, I encountered an error processing your request.", "sources": []}
        
        context = "\n".join([doc.page_content for doc in response.get("source_documents", [])])
        result = response.get("result", "No response")
        
        # Validate result is a string
        if not isinstance(result, str):
            result = str(result)
            logger.warning(f"Result was not a string, converted to: {result}")

        if is_context_sufficient(user_query, context):
            return {"response": result, "sources": response.get("source_documents", [])}

        # Step 2: If not sufficient, get definition from Tavily
        definition = tavily_lookup(user_query)
        logger.info(f"Definition: {definition}")
        if not definition:
            return {"response": "Sorry, I couldn't find information about that term.", "sources": []}

        rewritten_query = rewrite_query(user_query, definition)
        if not rewritten_query:
            return {"response": "Sorry, I couldn't find information about that term.", "sources": []}
        
        final_response = qa_chain.invoke({"query": rewritten_query})
        
        # Validate final response
        if not final_response or not isinstance(final_response, dict):
            logger.error("Invalid final response from qa_chain")
            return {"response": "Sorry, I encountered an error processing your request.", "sources": []}
        
        final_result = final_response.get("result", "No response")
        if not isinstance(final_result, str):
            final_result = str(final_result)
            logger.warning(f"Final result was not a string, converted to: {final_result}")

        return {"response": final_result, "sources": final_response.get("source_documents", [])}
        
    except Exception as e:
        logger.error(f"Error in agentic_rag_pipeline: {e}")
        return {"response": "Sorry, I encountered an error processing your request.", "sources": []}


if __name__ == "__main__":
    from app.components.create_index import create_index
    file_name="NASDAQ_AAPL_2024"
    create_index(file_name=file_name)
    result = agentic_rag_pipeline("What is percentage increase or decrease in operating income?", file_name="NASDAQ_AAPL_2024")
    print(result["response"])