FROM ghcr.io/astral-sh/uv:0.12.24-python3.13-trixie-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# git lets run manifests record the commit of the mounted repository (decision D10).
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/auditor

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev

RUN useradd --create-home --uid 1000 auditor
USER auditor
# The mounted repository belongs to the host user, which git would otherwise refuse.
RUN git config --global --add safe.directory /work

# The repository is mounted here, so relative paths in commands and .env work as on the host.
WORKDIR /work
ENTRYPOINT ["auditor"]
