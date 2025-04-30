# model_training/Dockerfile
# Use a Python image compatible with your pyproject.toml.
# Using slim is good for size, but might need build tools.
# If you plan to use GPU on Chameleon, you'll need an NVIDIA CUDA base image (more complex).
# This Dockerfile is for a CPU-only training job.
FROM python:3.9-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 # Unbuffer Python output for better logging visibility

# Install system dependencies required by Python packages (like openstacksdk, torch)
# libffi-dev, libssl-dev are common for network/auth libs
# build-essential, gfortran are often needed for compiling packages like numpy, scipy, or parts of torch
# curl, git needed by the Poetry installer script and potentially by Poetry itself
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        git \
        libffi-dev \
        libssl-dev \
        build-essential \
        gfortran \
    && rm -rf /var/lib/apt/lists/* # Clean up apt cache to reduce image size

# Install Poetry globally using the official recommended script
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python -

# Add Poetry's bin directory to the PATH
ENV PATH="/opt/poetry/bin:$PATH" \
    POETRY_VIRTUALENVS_IN_PROJECT=true # Create virtual environment inside the project directory

# Set the working directory inside the container
WORKDIR /app

# Copy the Poetry project files (pyproject.toml and poetry.lock)
# Copy these first to leverage Docker's build cache. If these files
# (and thus dependencies) don't change, Docker won't re-run poetry install.
COPY pyproject.toml poetry.lock* ./

# Install dependencies using Poetry
# --no-dev: Don't install development dependencies
# --no-interaction: Don't ask questions
# --no-ansi: Disable ANSI output (good for logging)
# --sync: Syncs the environment with poetry.lock
RUN poetry install --no-dev --no-interaction --no-ansi --sync

# Copy the application code
# Copy the refactored scripts and modules into the container's working directory (/app)
COPY model_training/swift_data_loader.py ./model_training/
COPY model_training/model.py ./model_training/
COPY model_training/utils.py ./model_training/
COPY model_training/train_job.py ./model_training/
# If you have other helper files in model_training/, copy them too
# COPY model_training/<other_file>.py ./model_training/

# The command to run the training script will be specified in docker-compose.yaml
# when using `docker compose run`. No CMD needed here.