# Use an official Python runtime as a parent image
#checkov:skip=CKV_DOCKER_2:Drone ingest is a worker process and does not expose an HTTP health endpoint.
FROM python:3.11-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's source code
COPY . .

RUN useradd --create-home --shell /usr/sbin/nologin atlas && \
    chown -R atlas:atlas /app
USER atlas

# Command to run the application
RUN ls -la /app
CMD ["python", "drone_client.py"]
