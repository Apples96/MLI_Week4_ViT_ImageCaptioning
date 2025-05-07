FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY streamlit_app.py .
COPY models.py .

# Create directories for models if they don't exist
RUN mkdir -p models

# Environment variables
ENV MODEL_PATH=/app/models/best_image_caption_model.pt
ENV HF_REPO_ID=YOUR_HUGGINGFACE_USERNAME/image-captioning-model

# Expose port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]