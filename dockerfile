# Base image
FROM python:3.9-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    POETRY_VERSION=1.2.2

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl build-essential

# Install poetry
RUN pip install poetry==$POETRY_VERSION

# Copy project files to container
WORKDIR /app
# Copy application code to container
COPY . /app

# Install project dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Start the application
CMD ["uvicorn", "tmlc.app.app:app", "--host", "0.0.0.0", "--port", "80"]
