FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python 3.10 and system dependencies with mirror fallback
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && \
    # Try multiple mirror options for better connectivity
    { \
        echo "deb http://archive.ubuntu.com/ubuntu jammy main restricted universe multiverse"; \
        echo "deb http://archive.ubuntu.com/ubuntu jammy-updates main restricted universe multiverse"; \
        echo "deb http://archive.ubuntu.com/ubuntu jammy-security main restricted universe multiverse"; \
    } > /etc/apt/sources.list && \
    apt-get update -o Acquire::Retries=3 -o Acquire::http::Timeout="60" && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    curl \
    wget \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python commands
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip and install Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

# Install other Python dependencies
RUN python3 -m pip install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and temp files
RUN mkdir -p /app/models /app/temp /app/logs

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python3", "app.py"]
