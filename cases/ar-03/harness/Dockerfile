ARG BASE_IMAGE=python:3.13-slim-bookworm@sha256:a5a7017efc5969e00ae2b91c8eb0631049a569652aa4ffc4501769049ae45c72
FROM ${BASE_IMAGE}
USER root
ARG REQUIREMENTS=cases/ar-03/harness/requirements-linux-arm64.txt
COPY ${REQUIREMENTS} /opt/ar03/requirements.txt
COPY cases/ar-03/harness/constraints-linux-arm64.txt /opt/ar03/constraints.txt
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir -r /opt/ar03/requirements.txt -c /opt/ar03/constraints.txt
COPY runner/main.py /opt/runner/main.py
ENV PYTHONDONTWRITEBYTECODE=1 \
    AR03_DETECTOR_DEVICE=cpu \
    AR03_TABLE_MODEL_PATH=/assets/models/DocLayout-YOLO/doclayout_yolo_docstructbench_imgsz1024.pt \
    AR03_STRUCTURE_MODEL_DIR=/assets/models/SLANet_plus \
    AR03_OCR_MODEL_DIR=/assets/models/PP-OCRv6_small_rec \
    PADDLE_PDX_CACHE_HOME=/tmp/paddlex \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    YOLO_CONFIG_DIR=/tmp/yolo \
    XDG_CACHE_HOME=/tmp/cache \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /tmp
USER 10001:10001
ENTRYPOINT ["python", "-I", "/opt/runner/main.py"]
