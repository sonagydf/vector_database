FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app
COPY . .

# Install system dependencies (for SSL, GCS, FAISS)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip

# Recommended: use a requirements.txt for consistency
COPY requirements.txt .
RUN pip install -r requirements.txt

# Start the FastAPI server
CMD ["uvicorn", "vector_db:app", "--host", "0.0.0.0", "--port", "8000"]
