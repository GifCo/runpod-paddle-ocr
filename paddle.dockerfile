# Use a standard NVIDIA CUDA 11.8 runtime
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
WORKDIR /app

# Install PyTorch (matching CUDA 11.8)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Paddle GPU & RunPod SDK
RUN pip install --no-cache-dir paddlepaddle-gpu==2.6.1 -i https://mirror.baidu.com/pypi/simple
RUN pip install --no-cache-dir "paddleocr>=2.8.0" pillow numpy modelscope runpod

# Copy the serverless handler
COPY handler.py /app/handler.py

# Execute the RunPod worker
CMD ["python", "-u", "/app/handler.py"]