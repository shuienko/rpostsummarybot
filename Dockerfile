# Use multi-stage build to reduce final image size
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
    py3-pip \
    build-base \
    && pip install --upgrade pip

# Copy and install dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels

# Stage 2 - Create minimal runtime image
FROM python:3.11-alpine

WORKDIR /app

# Install runtime dependencies
RUN apk add --no-cache libffi openssl

# Copy compiled wheels from builder stage
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy the bot code
COPY bot.py .

# Create volume mount point for .env file
VOLUME /app/config

# Set environment variable to point to the mounted .env file
ENV PYTHONUNBUFFERED=1
ENV DOTENV_PATH=/app/config/.env

# Run the bot
CMD ["python", "bot.py"]
