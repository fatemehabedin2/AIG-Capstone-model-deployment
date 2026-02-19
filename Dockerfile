FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code + model artifact
COPY app ./app
COPY models ./models

# Expose port (Cloud Run uses 8080 by default, but local is fine too)
EXPOSE 8080

# Cloud Run expects the app to listen on $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
