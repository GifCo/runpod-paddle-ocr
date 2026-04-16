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

# Install PaddlePaddle 3.3.x GPU (required for PaddleOCRVL / PaddleX pipeline)
RUN pip install --no-cache-dir paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Install PaddleOCR, paddlex[ocr] extras (required for PaddleOCR-VL-1.5 pipeline), & RunPod SDK
RUN pip install --no-cache-dir "paddleocr>=2.8.0" "paddlex[ocr]" pillow numpy modelscope runpod

# Copy the serverless handler
COPY handler.py /app/handler.py

# Execute the RunPod worker
CMD ["python", "-u", "/app/handler.py"]