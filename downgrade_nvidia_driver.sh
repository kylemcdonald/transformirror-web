#!/bin/bash
# Script to downgrade NVIDIA driver from 580.95.05 to 575.64.03

set -e  # Exit on error

echo "=========================================="
echo "NVIDIA Driver Downgrade Script"
echo "Current: 580.95.05"
echo "Target:  575.64.03"
echo "=========================================="
echo ""

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Please run this script with sudo"
    exit 1
fi

# Show current driver version (if available)
echo "Checking current driver version:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=driver_version --format=csv,noheader
else
    echo "nvidia-smi not found (driver may already be removed or not installed)"
fi
echo ""

# Update package list
echo "Updating package list..."
apt-get update

# Remove current driver (580 series)
echo ""
echo "Removing nvidia-driver-580..."
apt-get remove --purge -y nvidia-driver-580 nvidia-dkms-580 nvidia-kernel-source-580 || true

# Remove all NVIDIA packages to avoid conflicts
echo ""
echo "Cleaning up NVIDIA packages..."
apt-get autoremove -y

# Install specific version 575.64.03
echo ""
echo "Installing nvidia-driver-575 version 575.64.03..."

VERSION_575="575.64.03-0ubuntu0~gpu24.04.1"

# Update package cache again to ensure we have latest info
echo "Updating package cache..."
apt-get update

# Check if graphics-drivers PPA has the version
if apt-cache show nvidia-driver-575=$VERSION_575 2>/dev/null | grep -q "Version:"; then
    echo "Installing from graphics-drivers PPA (recommended)..."
    echo "Temporarily pinning PPA to higher priority to override CUDA repo..."
    
    # Create temporary apt preferences file to give PPA higher priority
    PREF_FILE="/etc/apt/preferences.d/99-temp-graphics-drivers-ppa"
    cat > "$PREF_FILE" <<EOF
Package: nvidia-*
Pin: origin ppa.launchpadcontent.net
Pin-Priority: 1000

Package: libnvidia-*
Pin: origin ppa.launchpadcontent.net
Pin-Priority: 1000
EOF
    
    # Update package cache with new preferences
    apt-get update
    
    echo "Installing all required packages from PPA to avoid dependency conflicts..."
    
    # Install all dependencies explicitly from PPA version to avoid conflicts
    # Use --allow-downgrades and --allow-change-held-packages to handle version conflicts
    apt-get install -y --allow-downgrades --allow-change-held-packages \
        libnvidia-gl-575=$VERSION_575 \
        nvidia-dkms-575=$VERSION_575 \
        nvidia-kernel-common-575=$VERSION_575 \
        nvidia-kernel-source-575=$VERSION_575 \
        libnvidia-compute-575=$VERSION_575 \
        libnvidia-extra-575=$VERSION_575 \
        nvidia-compute-utils-575=$VERSION_575 \
        libnvidia-decode-575=$VERSION_575 \
        libnvidia-encode-575=$VERSION_575 \
        nvidia-utils-575=$VERSION_575 \
        xserver-xorg-video-nvidia-575=$VERSION_575 \
        libnvidia-cfg1-575=$VERSION_575 \
        libnvidia-fbc1-575=$VERSION_575 \
        nvidia-driver-575=$VERSION_575 \
        nvidia-settings \
        nvidia-prime
    
    # Remove temporary preferences file
    echo "Cleaning up temporary apt preferences..."
    rm -f "$PREF_FILE"
    apt-get update
else
    echo "ERROR: PPA version $VERSION_575 not found!"
    echo "Checking available versions..."
    apt-cache madison nvidia-driver-575 | head -5
    exit 1
fi

echo ""
echo "=========================================="
echo "Driver installation completed!"
echo "=========================================="
echo ""
echo "IMPORTANT: You need to reboot your system for the changes to take effect."
echo ""
echo "After reboot, verify with: nvidia-smi"
echo ""

