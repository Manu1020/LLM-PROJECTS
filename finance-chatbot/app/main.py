from flask import Flask, render_template, request, session, redirect, url_for
from app.components.embeddings import get_embedding_model
from app.components.retriever import build_qa_chain
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
        try:
            # Index the selected company's documents
            create_index_for_company(company)
            session["selected_company"] = company
            session["messages"] = []  # Clear previous chat
            session["indexing_status"] = {
                "success": True,
                "message": f"✅ Successfully indexed {company} documents!"
            }
            session.pop("error", None)  # Clear any previous errors
        except Exception as e:
            session["indexing_status"] = {
                "success": False,
                "message": f"❌ Error indexing {company}: {str(e)}"
            }
            logger.error(f"Error indexing {company}: {str(e)}")
    return redirect(url_for("index"))

@app.route("/chat", methods=["POST"])
def chat():
    if "selected_company" not in session:
        session["error"] = "Please select a company first!"
        return redirect(url_for("index"))
    
    user_input = request.form.get("prompt")
    if user_input:
        if "messages" not in session:
            session["messages"] = []
        
        
        messages = session["messages"]
        messages.append({"role": "user", "content": user_input})
        session["messages"] = messages

        try:
            qa_chain = build_qa_chain(file_name=session["selected_company"])
            response = qa_chain.invoke({"query": user_input})
            logger.info(f"Response: {response}")
            result = response.get("result", "No response")
            messages.append({"role": "assistant", "content": result})
            session["messages"] = messages
            session.pop("error", None)  # Clear any previous errors
        except Exception as e:
            error_message = f"Error: {str(e)}"
            session["error"] = error_message
            logger.error(f"Error in chat: {str(e)}")
    
    return redirect(url_for("index"))

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
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)