# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /mnt/locust

# Copy the requirements file into the container at /mnt/locust
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Locust test file into the container at /mnt/locust
COPY locustfile.py .

# Default command to run when starting the container
# This will run Locust in master mode, exposing the web UI on port 8089.
# The locustfile is specified, and it will target services based on their internal Docker Compose hostnames.
ENTRYPOINT ["locust"]
CMD ["-f", "/mnt/locust/locustfile.py"]

# Expose the default Locust web UI port
EXPOSE 8089 