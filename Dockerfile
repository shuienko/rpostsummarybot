# Stage 1 - Build dependencies
FROM python:3.11-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    openssl-dev \
    python3-dev \
    build-base \
    && pip install --upgrade pip wheel

# Copy requirements and build wheels
COPY requirements.txt .
RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels

# Stage 2 - Minimal runtime image
FROM python:3.11-alpine

WORKDIR /app

# Install runtime dependencies
RUN apk add --no-cache libffi openssl

# Copy wheels from the builder stage
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy the bot code
COPY bot.py .

# Create volume mount point for .env file
VOLUME /app/config

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DOTENV_PATH=/app/config/.env

# Run the bot
CMD ["python", "bot.py"]
