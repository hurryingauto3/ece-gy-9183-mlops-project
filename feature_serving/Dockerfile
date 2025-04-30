# feature_serving/Dockerfile
# Use a Python image compatible with your pyproject.toml.
# Using slim is good for size, but might need build tools.
FROM python:3.9-slim

# Set environment variables for non-interactive apt install and unbuffered python output
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install system dependencies required by Python packages (like openstacksdk, uvicorn) and Poetry
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

# Copy the application code
# Copy the refactored scripts and modules into the container's working directory (/app)
COPY config.py ./feature_serving/
COPY data_loader.py ./feature_serving/
COPY main.py ./feature_serving/
COPY models.py ./feature_serving/
# If you have other helper files in feature_serving/, copy them too
# COPY feature_serving/<other_file>.py ./feature_serving/


# Expose the port the application will listen on (default uvicorn port 8001)
EXPOSE 8001

# Define the command to run when the container starts
# Use the entrypoint script managed by Poetry to run the uvicorn module
# This ensures uvicorn runs within the Poetry-managed virtual environment
# Command: poetry run uvicorn feature_serving.main:app --host 0.0.0.0 --port 8001
CMD ["poetry", "run", "uvicorn", "feature_serving.main:app", "--host", "0.0.0.0", "--port", "8001"]

# Alternative CMD if you prefer not to use `poetry run`
# This requires the poetry bin directory to be in the PATH
# CMD ["uvicorn", "feature_serving.main:app", "--host", "0.0.0.0", "--port", "8001"]
# The `poetry run` version is generally more robust as it guarantees the environment.