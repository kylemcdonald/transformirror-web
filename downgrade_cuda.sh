#!/bin/bash
# Script to downgrade CUDA from 13.0.88 to 12.8.93

set -e  # Exit on error

echo "=========================================="
echo "CUDA Downgrade Script"
echo "Current: 13.0.88"
echo "Target:  12.8.93"
echo "=========================================="
echo ""

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Please run this script with sudo"
    exit 1
fi

# Show current CUDA version
echo "Checking current CUDA version:"
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo "nvcc not found"
fi
echo ""

# Update package list
echo "Updating package list..."
apt-get update

# Remove all CUDA 13.0 packages
echo ""
echo "Removing CUDA 13.0 packages..."
# Get all installed CUDA 13.0 packages and remove them
dpkg -l | grep -E "cuda.*13" | awk '{print $2}' | xargs -r apt-get remove --purge -y || true

# Clean up
echo ""
echo "Cleaning up packages..."
apt-get autoremove -y
apt-get autoclean

# Update package cache again
echo ""
echo "Updating package cache..."
apt-get update

# Install CUDA 12.8 with specific nvcc version 12.8.93
echo ""
echo "Installing CUDA 12.8 with nvcc version 12.8.93-1..."

# First install the specific nvcc version and its dependencies
apt-get install -y \
    cuda-nvcc-12-8=12.8.93-1 \
    cuda-nvvm-12-8=12.8.93-1 \
    cuda-crt-12-8=12.8.93-1

# Then install the full CUDA 12.8 toolkit
apt-get install -y \
    cuda-toolkit-12-8

# Hold the specific nvcc version to prevent upgrades
echo ""
echo "Holding nvcc version 12.8.93-1 to prevent upgrades..."
apt-mark hold cuda-nvcc-12-8 cuda-nvvm-12-8 cuda-crt-12-8

# Update alternatives to use CUDA 12.8
echo ""
echo "Updating CUDA alternatives..."
update-alternatives --auto cuda
update-alternatives --auto cuda-12

# Verify installation
echo ""
echo "=========================================="
echo "CUDA downgrade completed!"
echo "=========================================="
echo ""
echo "Verifying installation..."
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
    echo ""
    echo "CUDA path: $(which nvcc)"
    echo "CUDA home: $(readlink -f /usr/local/cuda)"
else
    echo "WARNING: nvcc not found in PATH. You may need to:"
    echo "  export PATH=/usr/local/cuda/bin:\$PATH"
    echo "  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
fi
echo ""
echo "Note: If nvcc version doesn't show 12.8.93, you may need to:"
echo "  1. Log out and log back in (or restart terminal)"
echo "  2. Check your PATH and LD_LIBRARY_PATH environment variables"
echo ""

