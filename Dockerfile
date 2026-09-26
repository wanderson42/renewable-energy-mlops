FROM python:3.14-slim

# Configurações do Python e do Poetry
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONPATH="/app/src"

# Desativa alertas interativos do apt-get (remove os erros de debconf)
ARG DEBIAN_FRONTEND=noninteractive

# Instalação de dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir poetry

# Copia apenas os arquivos de configuração para usar o cache do Docker
COPY pyproject.toml poetry.lock ./

# Instala apenas as dependências (--no-root evita o erro do README.md)
RUN poetry install --only main --no-root --no-ansi

# Copia o restante do código fonte (incluindo src, README.md, etc.)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "energy_mlops.service.main:app", "--host", "0.0.0.0", "--port", "8000"]