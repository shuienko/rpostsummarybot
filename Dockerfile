# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the bot code
COPY bot.py .

# Create volume mount point for .env file
VOLUME /app/config

# Set environment variable to point to the mounted .env file
ENV PYTHONUNBUFFERED=1
ENV DOTENV_PATH=/app/config/.env

# Run the bot
CMD ["python", "bot.py"]