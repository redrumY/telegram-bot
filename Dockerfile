FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry
RUN poetry config virtualenvs.create false && poetry install --only main --no-interaction --no-ansi

# Copy application
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Set environment variables from .env if exists
ENV DATABASE_PATH=/app/data/memory.db

# Run the bot
CMD ["python", "main.py"]
