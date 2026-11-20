FROM python:3.13-slim-bookworm@sha256:a5a7017efc5969e00ae2b91c8eb0631049a569652aa4ffc4501769049ae45c72
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/opt/arena PATH=/opt/codex/codex-path:/opt/arena/.venv/bin:$PATH
WORKDIR /opt/arena
COPY pyproject.toml uv.lock ./
# Same runtime and researcher layout as Arena's Dockerfile; no GHCR dependency.
RUN pip install --no-cache-dir uv==0.8.13 && uv sync --frozen --no-dev --no-install-project \
    && useradd --uid 10001 --create-home arena \
    && apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates iptables squid \
    && rm -rf /var/lib/apt/lists/*
ARG CODEX_VERSION=0.170.0
RUN --mount=type=cache,target=/var/cache/ar03 \
    (test -f /var/cache/ar03/codex-${CODEX_VERSION}.tgz || \
     (curl -fL --retry 3 --retry-all-errors https://registry.npmjs.org/@openai/codex/-/codex-${CODEX_VERSION}-linux-arm64.tgz -o /var/cache/ar03/codex.pending \
      && mv /var/cache/ar03/codex.pending /var/cache/ar03/codex-${CODEX_VERSION}.tgz)) \
    && tar xzf /var/cache/ar03/codex-${CODEX_VERSION}.tgz -C /tmp \
    && python -c 'from pathlib import Path; import shutil; p=[p for p in Path("/tmp/package").rglob("codex") if p.is_file() and p.open("rb").read(4)==b"\x7fELF"]; assert len(p)==1,p; shutil.copytree(p[0].parent.parent,"/opt/codex"); Path("/usr/local/bin/codex").symlink_to("/opt/codex/bin/codex")' \
    && rm -r /tmp/package
COPY evaluator ./evaluator
COPY common ./common
COPY prompts ./prompts
COPY README.md AGENTS.md ./
COPY cases/ar-03/deploy/network-guard.sh cases/ar-03/deploy/squid.conf /opt/network/
COPY cases/ar-03/deploy/relay.py /opt/network/relay.py
USER 10001:10001
WORKDIR /workspace
CMD ["sleep", "infinity"]
