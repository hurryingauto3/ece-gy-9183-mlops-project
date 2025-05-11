FROM python:3.11.9

# Set working directory
WORKDIR /app

# Python environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Poetry setup
ENV POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    PATH="/opt/poetry/bin:$PATH"

RUN apt-get update || true \
    && apt-get install -y --no-install-recommends \
        curl \
        git \
        libffi-dev \
        libssl-dev \
        build-essential \
        gfortran \
        gnupg \
        ca-certificates \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 0xA1BCB2D2 || true \
    && rm -rf /var/lib/apt/lists/* # Clean up apt cache

# Install Poetry globally, specifying a version that supports package-mode (>=1.5.0)
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python - --version 2.1.1
COPY model_serving/pyproject.toml model_serving/poetry.lock ./
RUN poetry install --no-interaction --no-ansi --no-root

# --- Additions for dummy asset generation ---
# Create a dedicated directory for the asset generation script and its model definition dependency
RUN mkdir -p /app/asset_generator_tools

# Copy the specific model definitions file needed by the generation script
# This should be the model.py that ONLY contains class definitions, not training scripts.
COPY ../model_training/model.py /app/asset_generator_tools/model.py

# Copy the asset generation script (which now uses "from model import ...")
COPY ../model_training/generate_dummy_assets.py /app/asset_generator_tools/generate_assets.py

# Ensure the target directory for generated assets exists 
# (This is where the generate_assets.py script will write to)
RUN mkdir -p /app/dummy_assets && chmod 755 /app/dummy_assets
# --- End Additions ---

# Copy the rest of the model_serving application code
# This should only copy the model_serving specific files.
# Assuming your Dockerfile is in model_serving/ and context is the project root (.) 
# you might want to be more specific to avoid copying unnecessary parent directories.
# For now, if context is '.', COPY . . copies everything. 
# If context for this Dockerfile was model_serving/, then COPY . . would copy model_serving contents.
# Given docker-compose.dummy.yml uses context: ., the COPY . . copies the whole project. 
# This is a general Docker best practice point, but for now, the above specific copies should isolate the asset generation.
COPY . .

# Copy and set permissions for the entrypoint script
COPY ../model_serving/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose the port the app runs on
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Set the default command to run the application (will be passed to entrypoint.sh)
CMD ["uvicorn", "model_serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]