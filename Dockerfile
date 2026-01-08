# Use an NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables to make the installation non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and pip
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3-pip && \
    apt-get clean

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./atlas_brain ./atlas_brain

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the app.
# This will be executed when the container starts.
CMD ["uvicorn", "atlas_brain.main:app", "--host", "0.0.0.0", "--port", "8000"]
