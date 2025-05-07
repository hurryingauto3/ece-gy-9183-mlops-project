# model_training/Dockerfile
# Use a Python image compatible with your pyproject.toml.
# Using slim is good for size, but might need build tools.
# If planning to use GPU on Chameleon, you'll need an NVIDIA CUDA base image (more complex).
# This Dockerfile is for a CPU-only training job.
FROM python:3.10-slim

# Set environment variables for non-interactive apt install and unbuffered python output
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install system dependencies required by Python packages (like openstacksdk, torch) and Poetry
# libffi-dev, libssl-dev are common for network/auth libs
# build-essential, gfortran are often needed for compiling packages like numpy, scipy, or parts of torch
# curl, git needed by the Poetry installer script and potentially by Poetry itself
# Example RUN apt-get install line (adjust as needed based on other deps)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        git \
        libffi-dev \
        libssl-dev \
        build-essential \
        gfortran \
    && rm -rf /var/lib/apt/lists/* # Clean up apt cache

# Install Poetry globally, specifying a version that supports package-mode (>=1.5.0)
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python - --version 2.1.1

# Add Poetry's bin directory to the PATH and configure virtual environments
ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH" \
    POETRY_VIRTUALENVS_IN_PROJECT=true


# Set the working directory inside the container
WORKDIR /app

# Copy the Poetry project files (pyproject.toml and poetry.lock)
# Copy these first to leverage Docker's build cache. If these files
# (and thus dependencies) don't change, Docker won't re-run poetry install.
COPY pyproject.toml poetry.lock* ./

# Install dependencies using Poetry
# --no-dev: Don't install development dependencies
# --no-interaction: Don't ask questions (e.g., about virtual environments)
# --no-ansi: Disable ANSI output (good for logging)
# --sync: Syncs the environment with poetry.lock
RUN poetry install --no-interaction --no-ansi --no-root

# Copy the application code
# Copy the refactored scripts and modules into the container's working directory (/app)
COPY swift_data_loader.py ./model_training/
COPY model.py ./model_training/
COPY utils.py ./model_training/
COPY train_job.py ./model_training/
# If you have other helper files in model_training/, copy them too
# COPY model_training/<other_file>.py ./model_training/

# The command to run the training script will be specified in docker-compose.yaml
# when using `docker compose run`. No CMD is set here for this job image.
# CMD ["poetry", "run", "python", "model_training/train_job.py"] # Example CMD if you wanted a default