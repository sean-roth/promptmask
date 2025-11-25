FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and git (for pip git installs)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directory and set permissions for HF Spaces
RUN mkdir -p /app/output && chmod 777 /app/output

# Create a non-root user (required for HF Spaces)
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="7860"

WORKDIR /app

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
