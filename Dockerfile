FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY knowledge ./knowledge
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"
RUN useradd -u 10001 -M app && mkdir -p /app/data /app/results && chown -R app /app/data /app/results
USER 10001

EXPOSE 8000
CMD ["uvicorn", "iot_ai.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
