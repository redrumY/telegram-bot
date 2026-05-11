FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
COPY pyproject.toml poetry.lock ./
RUN pip install poetry --no-root-dir
RUN poetry config virtualenvs.create false && poetry install

# Copy application
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Set environment variables from .env if exists
ENV DATABASE_PATH=/app/data/memory.db

# Run the bot
CMD ["python", "main.py"]
