# Multi-stage build ottimizzato per salad.com
# Stage 1: Builder per dipendenze Python
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as builder

# Metadata
LABEL maintainer="DualMe Team"
LABEL description="DualMe Virtual Try-On - AI-powered clothing try-on system"
LABEL version="1.0.0"

# Evita prompts interattivi
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    unzip \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-glog-dev \
    libgflags-dev \
    libprotobuf-dev \
    protobuf-compiler \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Crea utente non-root per sicurezza
RUN useradd -m -u 1000 dualme && \
    mkdir -p /workspace && \
    chown -R dualme:dualme /workspace

# Imposta Python 3.10 come default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Aggiorna pip
RUN python -m pip install --upgrade pip setuptools wheel

# Installa PyTorch con CUDA support
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Stage 2: Applicazione finale
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Copia binari e librerie necessarie dallo stage builder
COPY --from=builder /usr/bin/python* /usr/bin/
COPY --from=builder /usr/lib/python3.10 /usr/lib/python3.10
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /etc/group /etc/group

# Installa runtime dependencies minime
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-glog0v5 \
    libgflags2.2 \
    libprotobuf23 \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Variabili d'ambiente per ottimizzazione GPU
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_MODULE_LOADING=LAZY
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

# Ottimizzazioni performance
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV TRANSFORMERS_CACHE=/workspace/cache/transformers
ENV HF_HOME=/workspace/cache/huggingface

# Working directory
WORKDIR /workspace

# Cambia all'utente non-root
USER dualme

# Aggiorna pip per l'utente
RUN python -m pip install --user --upgrade pip

# Copia requirements
COPY --chown=dualme:dualme requirements.txt .

# Installa dipendenze Python
RUN pip install --user --no-cache-dir -r requirements.txt

# Installa dipendenze aggiuntive per deployment
RUN pip install --user --no-cache-dir \
    gunicorn[gevent]==21.2.0 \
    uvloop==0.19.0 \
    httptools==0.6.1

# Copia codice applicazione
COPY --chown=dualme:dualme . .

# Rendi eseguibili gli script
RUN chmod +x download_models.py

# Crea directories necessarie
RUN mkdir -p ./ckpts ./data ./logs ./cache/transformers ./cache/huggingface /tmp/numba_cache

# Scarica modelli (opzionale - può essere fatto a runtime)
# RUN python download_models.py

# Esponi porta
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Script di avvio
COPY --chown=dualme:dualme <<EOF /workspace/start.sh
#!/bin/bash
set -e

echo "🚀 Starting DualMe Virtual Try-On..."

# Controlla se i modelli esistono, altrimenti scaricali
if [ ! -d "./ckpts/stable-diffusion-inpainting" ]; then
    echo "📦 Downloading models..."
    python download_models.py
fi

# Controlla GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🔥 GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo "⚠️ No GPU detected, running on CPU"
fi

# Avvia l'applicazione
echo "🌐 Starting Gradio server..."
exec python app_gradio.py
EOF

RUN chmod +x /workspace/start.sh

# Aggiungi endpoint di health check
COPY --chown=dualme:dualme <<EOF /workspace/health_check.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import torch
import psutil
import os

app = FastAPI()

@app.get("/health")
async def health_check():
    try:
        # Check GPU
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        # Check memory
        memory = psutil.virtual_memory()
        
        # Check disk space
        disk = psutil.disk_usage('/')
        
        status = {
            "status": "healthy",
            "gpu_available": gpu_available,
            "gpu_count": gpu_count,
            "memory_usage_percent": memory.percent,
            "disk_usage_percent": (disk.used / disk.total) * 100,
            "models_loaded": os.path.exists("./ckpts/stable-diffusion-inpainting")
        }
        
        return JSONResponse(content=status)
        
    except Exception as e:
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)}, 
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Comando di avvio predefinito
CMD ["/workspace/start.sh"]

# Metadata aggiuntive per salad.com
LABEL salad.gpu="required"
LABEL salad.min_gpu_memory="8GB" 
LABEL salad.recommended_gpu="RTX 3080"
LABEL salad.port="7860" 