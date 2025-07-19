# Use a slim Python image
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (modern Python package manager)
RUN pip install uv

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy the rest of your application code
COPY . .

# Expose the port your Flask app runs on
EXPOSE 5001

# Run the Flask app
CMD ["uv", "run", "python", "-m", "app.main"]

# # Run with Gunicorn for production
# CMD ["uv", "run", "gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "--timeout", "120", "--preload", "app.main:app"]