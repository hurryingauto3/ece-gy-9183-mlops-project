# Use a lightweight Python image based on Debian Buster (stable)
# Choose the Python version that matches your pyproject.toml
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install build dependencies for Poetry and potentially some Python packages
# Install curl or wget if needed for fetching poetry installation script
# Also install git as poetry might need it for some package sources (less likely for standard deps)
# Note: These are BUILD dependencies and can often be removed in a multi-stage build
# For simplicity, we'll keep them in this single stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
# Use the official recommended installer script
RUN curl -sSL https://install.python-poetry.org | python -

# Add Poetry to the PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy the project files into the container
# Copy pyproject.toml and poetry.lock first to leverage Docker's build cache
# If these files (and dependencies) don't change, this layer will be cached.
COPY pyproject.toml poetry.lock* /app/

# Install project dependencies using Poetry
# --no-dev: Do not install development dependencies
# --no-root: Do not install the root package itself (unless you make your service a package)
# Use poetry.lock to ensure deterministic dependency versions
RUN poetry install --no-dev --no-root

# Copy the rest of the application code
COPY ./feature_service /app/feature_service

# --- Environment Variables ---
# Declare expected environment variables with placeholder values
# These should NOT contain actual credentials here!
# They must be passed when running the container (e.g., -e OS_AUTH_URL=...)
ENV FS_OPENSTACK_AUTH_URL="http://<your-keystone-url>"
ENV FS_OPENSTACK_PROJECT_NAME="<your-project>"
ENV FS_OPENSTACK_PROJECT_DOMAIN_NAME="Default" # Set default if applicable
ENV FS_OPENSTACK_USERNAME="<your-username>"
ENV FS_OPENSTACK_USER_DOMAIN_NAME="Default"   # Set default if applicable
ENV FS_OPENSTACK_PASSWORD="<your-password>"   # Sensitive! Pass at runtime.
ENV FS_OPENSTACK_REGION_NAME=""               # Optional
ENV FS_OPENSTACK_SWIFT_CONTAINER_NAME="<your-container>"

# Optional: Set the logger level via environment variable (default INFO in your code)
# ENV LOG_LEVEL="INFO"


# Expose the port the application will listen on (default uvicorn port 8001)
EXPOSE 8001

# Define the command to run when the container starts
# Use the entrypoint script managed by Poetry to run the uvicorn module
# This ensures uvicorn runs within the Poetry-managed virtual environment
# Command: uvicorn feature_service.main:app --host 0.0.0.0 --port 8001
CMD ["poetry", "run", "uvicorn", "feature_service.main:app", "--host", "0.0.0.0", "--port", "8001"]

# Alternative CMD if you prefer not to use `poetry run`
# This requires the poetry bin directory to be in the PATH
# CMD ["uvicorn", "feature_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
# The `poetry run` version is generally more robust as it guarantees the environment.