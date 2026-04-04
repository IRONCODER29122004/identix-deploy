FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY mongodb_utils.py .
COPY deepfake_detector.py .
COPY templates/ templates/
COPY static/ static/

# Create necessary directories
RUN mkdir -p data/uploads data/pipelines_frames data/pipelines_crops

# Expose port (HF Spaces will map this)
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Run the Flask app
CMD ["python", "app.py"]
