ARG PYTHON_IMAGE=python:3.14-slim
FROM ${PYTHON_IMAGE}
ARG HERMES_REVISION=86fa9e6ca1de7ee71132d2be0103f3e37d1fa872
ARG UV_VERSION=0.8.13
RUN apt-get update && apt-get install -y --no-install-recommends git curl ripgrep ca-certificates build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==${UV_VERSION}
COPY cases/sc71/harness/hermes/source-cache/ /opt/hermes-source-cache/
COPY integrations/hermes/install.sh /tmp/arena-install-hermes.sh
RUN sh /tmp/arena-install-hermes.sh "${HERMES_REVISION}" && rm /tmp/arena-install-hermes.sh
RUN ln -s /assets/kb /kb
WORKDIR /opt/arena
COPY pyproject.toml uv.lock /opt/arena/
RUN uv sync --frozen --no-dev --extra tracking
COPY evaluator /opt/arena/evaluator
COPY runner/main.py /opt/arena/runner/main.py
COPY integrations/hermes/*.py integrations/hermes/config.yaml /opt/arena/integrations/hermes/
RUN /opt/hermes/.venv/bin/python -m integrations.hermes.expose_dependencies
COPY cases/sc71/harness/hermes/*.py /opt/arena/cases/sc71/harness/hermes/
RUN ln -sfn /opt/arena/integrations/hermes/config.yaml /opt/arena/cases/sc71/harness/hermes/config.yaml
ENV PATH="/opt/hermes/.venv/bin:${PATH}"
ENV PYTHONPATH="/opt/arena:/opt/hermes"
ENV HOME="/tmp/arena-home"
ENV HERMES_HOME="/tmp/arena-hermes-home"
ENV PYTHONDONTWRITEBYTECODE="1"
USER 10001:10001
ENTRYPOINT ["python", "/opt/arena/cases/sc71/harness/hermes/main.py"]
