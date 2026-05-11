FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY data ./data

RUN uv sync --frozen --no-dev

ENV WATCHQUEST_DATA_DIR=/app/data
ENV PATH="/app/.venv/bin:$PATH"

CMD ["watchquest-mcp"]
