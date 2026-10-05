FROM python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

FROM base AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG POETRY_VERSION=2.4.3

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Poetry isolado das dependências que serão copiadas para o runtime.
RUN python -m venv /opt/poetry \
    && /opt/poetry/bin/pip install --no-cache-dir "poetry==${POETRY_VERSION}" \
    && python -m venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Preserva todas as dependências main e as versões do lockfile.
RUN /opt/poetry/bin/poetry install --only main --no-root --no-ansi \
    && /opt/venv/bin/python -m pip check

FROM base AS runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
COPY data/reference/bahia_wind_capacity_checkpoints.csv ./data/reference/bahia_wind_capacity_checkpoints.csv

RUN test -s /app/data/reference/bahia_wind_capacity_checkpoints.csv

EXPOSE 8000
CMD ["uvicorn", "energy_mlops.service.main:app", "--host", "0.0.0.0", "--port", "8000"]
