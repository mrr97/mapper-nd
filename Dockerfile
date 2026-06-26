# syntax=docker/dockerfile:1
FROM python:3.12-slim

# numpy / scikit-learn need the OpenMP runtime at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv: fast, reproducible installs from the committed lockfile.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Install dependencies as a cached layer, then the project and its bundled data.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
COPY data ./data

# Interactive Dash app.
EXPOSE 8050

CMD ["mapper-nd", "dashboard"]
