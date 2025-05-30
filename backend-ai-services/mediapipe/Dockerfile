FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# MediaPipe specific environment variables to suppress warnings  
ENV GLOG_minloglevel=3
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV ABSL_STDERR_THRESHOLD=3
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV GRPC_VERBOSITY=ERROR
ENV GRPC_TRACE=""
ENV GLOG_logtostderr=0
ENV GLOG_alsologtostderr=0
ENV TF_ENABLE_GPU_GARBAGE_COLLECTION=false

# GPU acceleration environment variables
ENV MEDIAPIPE_GPU_ALLOW_GROWTH=1
ENV MEDIAPIPE_GPU_MEMORY_FRACTION=0.4
ENV CUDA_CACHE_DISABLE=0
ENV DISPLAY=:0

# Configure apt for better reliability and use multiple mirrors
RUN echo "deb http://archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list

# Install system dependencies with virtual display support
RUN apt-get clean && \
    apt-get update -o Acquire::Retries=5 -o Acquire::http::Timeout="120" -o Acquire::https::Timeout="120" && \
    apt-get install -y --no-install-recommends --fix-missing \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-dev \
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
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install display-related packages separately to handle potential failures
RUN apt-get update -o Acquire::Retries=5 -o Acquire::http::Timeout="120" && \
    apt-get install -y --no-install-recommends --fix-missing \
    mesa-common-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    xvfb \
    x11-utils \
    mesa-utils \
    || echo "Some display packages failed to install, continuing with basic setup" && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Upgrade pip and install Python dependencies with retries
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir --timeout 600 --retries 10 -r requirements.txt || \
    (echo "Retrying with individual package installation..." && \
     python3 -m pip install --no-cache-dir mediapipe>=0.10.7 && \
     python3 -m pip install --no-cache-dir opencv-python-headless>=4.9.0 && \
     python3 -m pip install --no-cache-dir numpy>=1.24.0 && \
     python3 -m pip install --no-cache-dir Pillow>=10.0.0 && \
     python3 -m pip install --no-cache-dir fastapi>=0.104.0 && \
     python3 -m pip install --no-cache-dir "uvicorn[standard]>=0.24.0" && \
     python3 -m pip install --no-cache-dir pydantic>=2.4.0 && \
     python3 -m pip install --no-cache-dir python-multipart>=0.0.6 && \
     python3 -m pip install --no-cache-dir aiofiles>=23.0.0 && \
     python3 -m pip install --no-cache-dir httpx>=0.25.0 && \
     python3 -m pip install --no-cache-dir loguru>=0.7.0 && \
     python3 -m pip install --no-cache-dir prometheus-client>=0.18.0 && \
     python3 -m pip install --no-cache-dir psutil>=5.9.0 && \
     python3 -m pip install --no-cache-dir tensorflow>=2.15.0)

COPY . .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
