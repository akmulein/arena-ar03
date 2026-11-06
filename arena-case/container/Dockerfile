# syntax=docker/dockerfile:1
FROM python:3.13-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

FROM base AS runner
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng && rm -rf /var/lib/apt/lists/*
COPY container/runner-requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
COPY --from=arena_runner /main.py /opt/runner/main.py
USER 10001:10001
WORKDIR /tmp
ENTRYPOINT ["python", "-I", "/opt/runner/main.py"]

FROM base AS metric
COPY container/metric-requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
COPY metric /metric
USER 10001:10001
WORKDIR /tmp
ENTRYPOINT ["python", "-B", "/metric/score.py"]
