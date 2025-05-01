# Stage 1: Builder
FROM python:3.10-slim AS builder

# Install poetry
ENV POETRY_VERSION=1.7.1
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /build

# Copy only files needed for dependency installation
COPY pyproject.toml poetry.lock* /build/

# Install dependencies and export requirements
# --without dev: Exclude development dependencies like pytest from the export
# --output: Specify the output file name
# --without-hashes: Improves compatibility sometimes, consider if needed
RUN poetry export --without dev --format requirements.txt --output requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

# Create a non-root user and group
ARG USER=appuser
ARG GROUP=appgroup
ARG UID=1001
ARG GID=1001
RUN groupadd -g ${GID} ${GROUP} && \
    useradd -u ${UID} -g ${GROUP} -m -s /bin/bash ${USER}

WORKDIR /app

# Copy exported requirements from builder stage
COPY --from=builder /build/requirements.txt /app/requirements.txt

# Install runtime dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code from build context
COPY . /app/model_serving

# Set environment variables
ENV MLFLOW_TRACKING_URI=""
ENV FEATURE_SERVICE_URL=""
# Ensure this is passed during docker run
ENV PYTHONPATH=/app
# Add environment variable for number of workers (default to 2 if not set)
# ENV UVICORN_WORKERS=${UVICORN_WORKERS:-2}

# Change ownership of the app directory
RUN chown -R ${USER}:${GROUP} /app

# Switch to non-root user
USER ${USER}

EXPOSE 8000

# Adjust the CMD to point to the app within the model_serving directory
# Use the $UVICORN_WORKERS environment variable
CMD ["uvicorn", "model_serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
#  "--workers", "${UVICORN_WORKERS}"]