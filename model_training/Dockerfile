# model_training/Dockerfile
FROM python:3.10-slim

# non-interactive & unbuffered output
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# system deps for poetry, torch, openstacksdk, etc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         curl \
         git \
         libffi-dev \
         libssl-dev \
         build-essential \
         gfortran \
    && rm -rf /var/lib/apt/lists/*

# install Poetry (>=1.5.0)
RUN curl -sSL https://install.python-poetry.org \
       | POETRY_HOME=/opt/poetry python - --version 2.1.1

ENV POETRY_HOME="/opt/poetry" \
    PATH="$POETRY_HOME/bin:$PATH" \
    POETRY_VIRTUALENVS_IN_PROJECT=true

WORKDIR /app

# install Python deps
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --no-root

# copy all code
COPY . .

# prepare input/output mount points
RUN mkdir -p model_training/input model_training/output

# no default CMD; override with:
#   docker compose run model-training python model_training/train.py …
# or
#   docker compose run model-training python model_training/inference.py …