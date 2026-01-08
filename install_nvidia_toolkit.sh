#!/bin/bash
# This script installs the NVIDIA Container Toolkit.
# It must be run with sudo privileges.

set -e

echo "Step 1: Setting up the repository key..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

echo "Step 2: Adding the repository..."
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "Step 3: Updating package list..."
apt-get update

echo "Step 4: Installing the toolkit..."
apt-get install -y nvidia-container-toolkit

echo "Step 5: Restarting Docker..."
systemctl restart docker

echo "Installation complete. You can now try running 'docker compose up -d' again."
