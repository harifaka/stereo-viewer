FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# cpu keeps the image small. Use cu128 (with docker-compose.gpu.yml) for CUDA PyTorch,
# which accelerates YOLO pose/detection and TAPIR. OpenCV StereoSGBM stays on the CPU
# unless OpenCV itself is built with CUDA.
ARG TORCH_VARIANT=cpu
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    torch==2.8.0+${TORCH_VARIANT} torchvision==0.23.0+${TORCH_VARIANT}
RUN pip install --no-cache-dir -r requirements.txt \
    && (pip uninstall -y opencv-python opencv-python-headless || true) \
    && pip install --no-cache-dir opencv-contrib-python==4.8.1.78
# TAPIR (Google DeepMind TAP-Net family) for temporal point tracking. --no-deps avoids the
# JAX stack and keeps opencv-contrib-python; the PyTorch model only needs einshape and dm-tree.
ARG INSTALL_TAPNET=0
RUN if [ "$INSTALL_TAPNET" = "1" ]; then \
        pip install --no-cache-dir einshape dm-tree \
        && pip install --no-cache-dir --no-deps https://github.com/google-deepmind/tapnet/archive/refs/heads/main.zip; \
    fi
# Open3D enables Poisson surface reconstruction for mesh export.
ARG INSTALL_OPEN3D=0
RUN if [ "$INSTALL_OPEN3D" = "1" ]; then pip install --no-cache-dir open3d==0.18.0; fi
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
