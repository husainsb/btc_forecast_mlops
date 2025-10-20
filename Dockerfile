ARG PYTHON_VERSION=3.12.11
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Use official python image as a base
#FROM python:3.12.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libopenblas-dev git && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /model
WORKDIR /model

# Expose FastAPI default port
EXPOSE 8000

# Start FastAPI server using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]