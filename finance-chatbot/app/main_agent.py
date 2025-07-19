from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from app.components.embeddings import get_embedding_model
from app.components.agentic_rag import agentic_rag_pipeline
from app.components.create_index import create_index
from app.common.logger import get_logger
import os

logger = get_logger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

from markupsafe import Markup

def nl2br(text):
    return Markup(text.replace('\n', '\n'))

app.jinja_env.filters['nl2br'] = nl2br

@app.route("/", methods=["GET"])
def index():
    if "messages" not in session:
        session["messages"] = []
    
    return render_template("index.html", 
                         messages=session.get("messages", []),
                         indexing_status=session.get("indexing_status"),
                         error=session.get("error"))

@app.route("/index_documents", methods=["POST"])
def index_documents():
    company = request.form.get("company")
    if company:
        # Clear old indexing status and messages when selecting a new company
        old_company = session.get("selected_company")
        if old_company and old_company != company:
            # Clear old status and messages when switching companies
            session.pop("indexing_status", None)
            session.pop("messages", None)
            session.pop("sources", None)
        
        try:
            # Index the selected company's documents
            create_index_for_company(company)
            session["selected_company"] = company
            
            # Only clear messages if this is a new company or first time
            if old_company != company:
                session["messages"] = []
            
            session["indexing_status"] = {
                "success": True,
                "message": f"Successfully indexed {company} documents!"
            }
            session.pop("error", None)  # Clear any previous errors
        except Exception as e:
            session["indexing_status"] = {
                "success": False,
                "message": f"Error indexing {company}: {str(e)}"
            }
            logger.error(f"Error indexing {company}: {str(e)}")
    return redirect(url_for("index"))

@app.route("/chat", methods=["POST"])
def chat():
    if "selected_company" not in session:
        return jsonify({"success": False, "error": "Please select a company first!"})

    user_input = request.form.get("prompt")
    if not user_input:
        return jsonify({"success": False, "error": "No input provided"})

    try:
        result = agentic_rag_pipeline(user_input, file_name=session["selected_company"])
        response = result.get("response", "No response")
        source_documents = result.get("sources", [])

        # Format sources for JSON response
        sources = []
        for i, doc in enumerate(source_documents, 1):
            source_info = {
                "number": i,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": {
                    "page": doc.metadata.get("page")
                }
            }
            sources.append(source_info)

        return jsonify({
            "success": True,
            "response": response,
            "sources": sources
        })

    except Exception as e:
        error_message = f"Error: {str(e)}"
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({"success": False, "error": error_message})

# Add a new route to fetch sources for a specific message
@app.route("/get_sources/<message_index>")
def get_sources(message_index):
    sources = session.get("sources", {}).get(message_index, [])
    return {"sources": sources}


@app.route("/clear")
def clear():
    session.pop("messages", None)
    # session.pop("selected_company", None)
    # session.pop("indexing_status", None)
    session.pop("error", None)
    return redirect(url_for("index"))

def create_index_for_company(company):
    """Index documents for specific company"""
    try:
        logger.info(f"Indexing documents for company: {company}")
        create_index(file_name=company, force_reindex=False)
        logger.info(f"Successfully indexed documents for {company}")
    except Exception as e:
        logger.error(f"Error creating index for {company}: {str(e)}")
        raise e

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)