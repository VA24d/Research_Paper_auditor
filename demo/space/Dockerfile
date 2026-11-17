FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
COPY --chown=user pyproject.toml uv.lock README.md ./
COPY --chown=user src ./src
RUN uv sync --frozen --no-dev --extra ui
COPY --chown=user demo ./demo
ENV AUDITOR_DEMO=1 \
    AUDITOR_RUNS_DIR=demo/runs \
    AUDITOR_DATA_DIR=/tmp/auditor-data
EXPOSE 7860
CMD [".venv/bin/streamlit", "run", "src/auditor/ui/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
