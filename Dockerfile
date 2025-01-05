# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Environment variables will be provided via docker-compose or at runtime
ENV TELEGRAM_TOKEN=""
ENV REDDIT_CLIENT_ID=""
ENV REDDIT_CLIENT_SECRET=""
ENV OLLAMA_URL="http://localhost:11434/api/generate"
ENV OLLAMA_MODEL="llama3.2"

# Run the bot
CMD ["python", "bot.py"]