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
        logger.info(f"Rewritten query: {response.content}")
        return response.content
    except Exception as e:
        logger.error(f"Error rewriting query: {e}")
        return None

def agentic_rag_pipeline(user_query, file_name=None):
    # Step 1: Try to answer from vector DB
    qa_chain = build_qa_chain(file_name=file_name)
    response = qa_chain.invoke({"query": user_query})
    context = "\n".join([doc.page_content for doc in response.get("source_documents", [])])

    if is_context_sufficient(user_query, context):
        return {"response": response.get("result", "No response"), "sources": response.get("source_documents", [])}

    # Step 2: If not sufficient, get definition from Tavily
    definition = tavily_lookup(user_query)
    logger.info(f"Definition: {definition}")
    if not definition:
        return {"response": "Sorry, I couldn't find information about that term.", "sources": []}

    rewritten_query = rewrite_query(user_query, definition)
    # Step 3: Search vector DB again for examples/context using the definition
    if not rewritten_query:
        return {"response": "Sorry, I couldn't find information about that term.", "sources": []}
    
    final_response = qa_chain.invoke({"query": rewritten_query})

    return {"response": final_response.get("result", "No response"), "sources": final_response.get("source_documents", [])}


if __name__ == "__main__":
    from app.components.create_index import create_index
    file_name="NASDAQ_AAPL_2024"
    create_index(file_name=file_name)
    result = agentic_rag_pipeline("What is percentage increase or decrease in operating income?", file_name="NASDAQ_AAPL_2024")
    print(result["response"])
    # print(result["sources"])