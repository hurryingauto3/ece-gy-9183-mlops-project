# etl_job/Dockerfile
FROM python:3.9-slim

# Set environment variables to prevent interactive questions during apt install
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 # Unbuffer Python output for better logging visibility

# Install system dependencies required for openstacksdk (like SSL libs)
# and packages needed for Poetry and potential build steps (like curl, git)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        git \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/* # Clean up apt cache to reduce image size

# Install Poetry globally
RUN pip install poetry

# Add Poetry's bin directory to the PATH
ENV PATH="/root/.poetry/bin:$PATH"

# Set the working directory inside the container
WORKDIR /app

# Copy the Poetry project files (pyproject.toml and poetry.lock)
# Copy these first so Docker can cache this layer if dependencies don't change
# The build context is ./etl_job, so paths are relative to that.
COPY pyproject.toml poetry.lock* ./

# Install dependencies using Poetry
# --no-dev: Don't install dev dependencies
# --no-interaction: Don't ask questions
# --no-ansi: Disable ANSI output (good for logging)
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy the ETL script
# The build context is ./etl_job, so paths are relative to that.
COPY drive_to_swift_etl.py ./

# The command to run the script will be defined in docker-compose.yml
# We don't set CMD here.