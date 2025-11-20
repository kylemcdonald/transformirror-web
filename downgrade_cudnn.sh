#!/bin/bash
# Script to remove current cuDNN and downgrade to version 9.10.2

set -e  # Exit on error

echo "=========================================="
echo "cuDNN Downgrade Script"
echo "Target: 9.10.2"
echo "=========================================="
echo ""

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Please run this script with sudo"
    exit 1
fi

# Show current cuDNN version (if available)
echo "Checking current cuDNN installation:"
if dpkg -l | grep -q cudnn; then
    echo "Current cuDNN packages:"
    dpkg -l | grep cudnn
else
    echo "No cuDNN packages found in dpkg"
fi
echo ""

# Remove current cuDNN packages
echo "=========================================="
echo "Removing current cuDNN installation..."
echo "=========================================="

# Remove all cuDNN packages
apt-get remove --purge -y libcudnn* cudnn* || true

# Also remove any cuDNN packages that might be installed via different names
apt-get remove --purge -y \
    libcudnn8 \
    libcudnn8-dev \
    libcudnn9 \
    libcudnn9-dev \
    cudnn-local-repo-* \
    || true

# Clean up
echo ""
echo "Cleaning up package cache..."
apt-get autoremove -y
apt-get autoclean

# Remove any existing cuDNN repository files
echo ""
echo "Removing existing cuDNN repository files..."
rm -rf /var/cudnn-local-repo-* || true
rm -f /etc/apt/sources.list.d/cudnn-local-repo-*.list || true
rm -f /usr/share/keyrings/cudnn-*-keyring.gpg || true

echo ""
echo "=========================================="
echo "Installing cuDNN 9.10.2..."
echo "=========================================="

# Change to a temporary directory
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

# Download cuDNN 9.10.2
echo ""
echo "Downloading cuDNN 9.10.2 package..."
wget https://developer.download.nvidia.com/compute/cudnn/9.10.2/local_installers/cudnn-local-repo-ubuntu2404-9.10.2_1.0-1_amd64.deb

# Install the local repository package
echo ""
echo "Installing cuDNN local repository..."
dpkg -i cudnn-local-repo-ubuntu2404-9.10.2_1.0-1_amd64.deb

# Copy the keyring
echo ""
echo "Setting up GPG keyring..."
cp /var/cudnn-local-repo-ubuntu2404-9.10.2/cudnn-*-keyring.gpg /usr/share/keyrings/

# Update package list
echo ""
echo "Updating package list..."
apt-get update

# Install cuDNN
echo ""
echo "Installing cuDNN 9.10.2..."
apt-get -y install cudnn

# Clean up downloaded package
echo ""
echo "Cleaning up temporary files..."
cd /
rm -rf "$TMP_DIR"

echo ""
echo "=========================================="
echo "cuDNN downgrade completed!"
echo "=========================================="
echo ""
echo "Installed cuDNN packages:"
dpkg -l | grep cudnn
echo ""
echo "You can verify the installation by checking:"
echo "  dpkg -l | grep cudnn"
echo ""




