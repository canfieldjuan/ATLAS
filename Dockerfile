# Use an NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set environment variables to make the installation non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./atlas_brain ./atlas_brain

# Copy the extracted standalone packages so the host's api package
# can lazy-import them at startup (Content Ops control-surface
# routes are mounted in atlas_brain/api/__init__.py via
# `from extracted_content_pipeline.api.control_surfaces import ...`
# inside a try/except). When the packages are absent the import
# logs a warning and the route is skipped, but the prod image
# should ship them so the new screens are functional.
COPY ./extracted_content_pipeline ./extracted_content_pipeline
COPY ./extracted_quality_gate ./extracted_quality_gate
COPY ./extracted_reasoning_core ./extracted_reasoning_core
COPY ./extracted ./extracted

RUN useradd --create-home --shell /usr/sbin/nologin atlas && \
    chown -R atlas:atlas /app
USER atlas

# Expose the port the app runs on
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3.11 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/ping', timeout=5)" || exit 1

# Define the command to run the app.
# This will be executed when the container starts.
CMD ["uvicorn", "atlas_brain.main:app", "--host", "0.0.0.0", "--port", "8000"]
