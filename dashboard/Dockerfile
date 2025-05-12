FROM python:3.11.9

# Set working directory
WORKDIR /app

# Python environment variables
ENV PYTHONDONTWRITEBYTECODE 1 \
    PYTHONUNBUFFERED 1 \
    POETRY_NO_INTERACTION 1 # Ensure no interactive prompts from Poetry

# Install Poetry
ENV POETRY_VERSION=1.7.1
# It's often better to install poetry into a location that's already on PATH
# or ensure its bin is added. Pipx is good for this, but for simplicity:
RUN pip install "poetry==$POETRY_VERSION"

# Copy only files necessary for dependency installation first to leverage Docker cache
COPY pyproject.toml poetry.lock* ./

# Install dependencies using Poetry
# --no-root: Do not install the project itself as a package
# If streamlit is a main dependency, it should be installed here.
RUN poetry install --no-ansi --no-root

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application using poetry run
CMD ["poetry", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 