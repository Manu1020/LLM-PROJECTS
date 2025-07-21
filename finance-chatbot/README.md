# Finance Chatbot

An AI powered chatbot for answering questions about financial reports using Large Language Models (LLMs). This project enables users to interactively query financial reports (PDFs) of any public listed companies like Apple, Meta with features such as web-augmented definitions and conversational context.

---

## Features

- **Conversational Q&A**: Ask questions about financial reports in natural language.
- **Retrieval-Augmented Generation (RAG)**: Answers are grounded in the actual content of uploaded or indexed financial PDFs.
- **Agentic Pipeline**: For definition queries, the chatbot can augment answers with web search results and rewrite queries for clarity.
- **Source Attribution**: Responses include references to the source document and page.
- **Web UI**: Simple Flask-based web interface for chat and document management.
- **Modern Packaging**: Uses [uv](https://github.com/astral-sh/uv) for fast Python dependency management.
- **Docker Support**: Containerized for easy deployment.

---

## Project Structure

```
finance-chatbot/
  app/
    components/      # Core logic: RAG, agent, embeddings, retriever, etc.
    common/          # Logging and custom exceptions
    config/          # Configuration files
    templates/       # HTML templates for the web UI
    main_agent.py    # Agentic RAG Flask app entry point
  data/              # Financial report PDFs
  vector_db/         # Vector database files (auto-generated)
  requirements.txt   # Python dependencies (for reference)
  pyproject.toml     # Project metadata and dependencies
  uv.lock            # uv dependency lock file
  Dockerfile         # For containerized deployment
```

Let me know if you want to further customize any section!

---

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (for local development)
- (Optional) Docker

### Installation (Local)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Manu1020/llm-projects.git
   cd finance-chatbot
   ```

2. **Install dependencies with uv:**
   ```bash
   uv sync --frozen --no-dev
   ```

3. **Add your financial PDFs:**
   - Place your company financial reports (PDFs) in the `data/` directory.
   - Example files: `NASDAQ_AAPL_2024.pdf`, `NASDAQ_AMZN_2024.pdf`, etc.

4. **Set environment variables (if needed):**
   - You may need API keys for LLM providers (OpenAI, HuggingFace, etc.).
   - Create a `.env` file or export variables as needed.

---

## Running the App

```bash
uv run python -m app.main_agent
```

- Access the web UI at [http://localhost:5001](http://localhost:5001).

---

## Running with Docker and uv

This project uses [uv](https://github.com/astral-sh/uv) for fast, modern Python dependency management and packaging.

### Build the Docker Image

```bash
docker build -t finance-chatbot .
```

### Run the Container

```bash
docker run -p 5001:5001 finance-chatbot
```

- The app will be available at [http://localhost:5001](http://localhost:5001).

#### Notes

- Dependencies are managed with `uv` using `pyproject.toml` and `uv.lock`.
- The Dockerfile exposes port `5001` (update your run command and browser URL accordingly).
- For production, you can uncomment the Gunicorn command in the Dockerfile for better performance.

---

## Usage

1. Open the web interface.
2. Select a company to index its financial report.
3. Ask questions about the report (e.g., "What was the net income in 2024?").
4. For financial definitions, the agentic mode will augment answers with web search results.

---

## Supported Companies / Data

- Apple (AAPL)
- Amazon (AMZN)
- Meta (META)
- Microsoft (MSFT)
- Nvidia (NVDA)
- (Add more PDFs to `data/` as needed)

---

## Contributing

Contributions are welcome! Please open issues or pull requests for improvements or new features.

## Contact

For questions or support, please contact [maanasapriya.koduri@gmail.com].

---

*This README was generated based on the project structure and code as of June 2024.*
