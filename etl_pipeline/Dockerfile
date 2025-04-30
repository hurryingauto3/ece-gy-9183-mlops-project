# ETL/Dockerfile
FROM python:3.9-slim

# Set environment variables for non-interactive apt install and unbuffered python output
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install system dependencies required for openstacksdk and Poetry
# libffi-dev, libssl-dev are common for network/auth libs
# curl, git needed by the Poetry installer script
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        git \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/* # Clean up apt cache

# Install Poetry globally using the official recommended script
# Installs Poetry to /opt/poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python -

# Add Poetry's bin directory to the PATH and configure virtual environments
ENV PATH="/opt/poetry/bin:$PATH" \
    POETRY_VIRTUALENVS_IN_PROJECT=true

# Set the working directory inside the container
WORKDIR /app

# Copy the Poetry project files (pyproject.toml and poetry.lock)
# Copy these first to leverage Docker's build cache. If these files
# (and thus dependencies) don't change, Docker won't re-run poetry install.
COPY pyproject.toml poetry.lock* ./

# Install dependencies using Poetry
# --no-dev: Don't install development dependencies
# --no-interaction: Don't ask questions
# --no-ansi: Disable ANSI output
# --sync: Syncs the environment with poetry.lock
RUN poetry install --no-dev --no-interaction --no-ansi --sync

# Copy the ETL script
# The build context is ./ETL, so paths are relative to that.
COPY drive_to_swift_etl.py ./

# The command to run the script will be specified in docker-compose.yaml
# when using `docker compose run`. No CMD is set here for this job image.
# CMD ["poetry", "run", "python", "drive_to_swift_etl.py"] # Example CMD if you wanted a default