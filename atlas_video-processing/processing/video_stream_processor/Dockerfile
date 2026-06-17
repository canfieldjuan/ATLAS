# Use an official Python runtime as a parent image
#checkov:skip=CKV_DOCKER_2:Video stream processor is a worker process and does not expose an HTTP health endpoint.
FROM python:3.13-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libxext6 \
    libsm6 \
    libxrender1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's source code
COPY . .

RUN useradd --create-home --shell /usr/sbin/nologin atlas && \
    chown -R atlas:atlas /app
USER atlas

# Command to run the application
CMD ["python", "video_stream_processor.py"]
