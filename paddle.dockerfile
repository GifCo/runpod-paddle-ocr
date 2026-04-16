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

# Pre-download model weights at build time so they're baked into the image.
# Without this, ~2GB of weights download on every cold start.
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1
RUN python -c "\
import os, warnings, logging; \
os.environ['FLAGS_enable_pir_api']='1'; \
os.environ['FLAGS_use_mkldnn']='0'; \
warnings.filterwarnings('ignore'); \
logging.getLogger('ppocr').setLevel(logging.ERROR); \
import paddle; paddle.set_device('cpu'); \
from paddleocr import PaddleOCRVL; \
PaddleOCRVL(pipeline_version='v1.5', use_doc_orientation_classify=False, use_doc_unwarping=False); \
print('✅ Models cached successfully') \
"

# Copy the serverless handler
COPY handler.py /app/handler.py

# Execute the RunPod worker
CMD ["python", "-u", "/app/handler.py"]