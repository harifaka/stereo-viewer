FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# Use the CPU wheels so PyTorch does not pull in CUDA runtime libraries.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.8.0+cpu torchvision==0.23.0+cpu
RUN pip install --no-cache-dir -r requirements.txt \
    && (pip uninstall -y opencv-python opencv-python-headless || true) \
    && pip install --no-cache-dir opencv-contrib-python==4.8.1.78
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
