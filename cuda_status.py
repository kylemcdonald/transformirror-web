#!/usr/bin/env python3
"""
CUDA Status Check Script for RTX 5090
This script provides a comprehensive check of CUDA availability and versions.
"""

import torch
import subprocess
import os

def main():
    print("=== CUDA Status Check for RTX 5090 ===")
    print()
    
    # Check NVIDIA driver and GPU
    print("1. NVIDIA Driver and GPU:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version:' in line:
                    driver_version = line.split('Driver Version:')[1].strip().split()[0]
                    print(f"   Driver Version: {driver_version}")
                elif 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    print(f"   CUDA Version (Driver): {cuda_version}")
                elif 'RTX 5090' in line:
                    print(f"   GPU: RTX 5090 detected")
        else:
            print("   ❌ nvidia-smi failed")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Check CUDA toolkit
    print("\n2. CUDA Toolkit:")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'release' in line.lower():
                    print(f"   {line.strip()}")
        else:
            print("   ❌ nvcc not found")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Check PyTorch
    print("\n3. PyTorch:")
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            print(f"   Device {i}: {device_name}")
    else:
        print("   ❌ CUDA not available")
    
    # Check cuDNN
    print("\n4. cuDNN:")
    if torch.backends.cudnn.is_available():
        print(f"   Available: Yes")
        print(f"   Version: {torch.backends.cudnn.version()}")
    else:
        print("   Available: No")
    
    print("\n=== Summary ===")
    if torch.cuda.is_available():
        print("✅ CUDA is working correctly!")
        print("You can use GPU acceleration with PyTorch.")
    else:
        print("❌ CUDA is not working.")
        print("\nPossible issues:")
        print("1. CUDA version mismatch between driver (12.9) and toolkit (12.8)")
        print("2. PyTorch compiled with CUDA 12.8 but driver supports 12.9")
        print("3. CUDA context initialization error")
        print("\nRecommended solutions:")
        print("1. Restart your system to clear CUDA contexts")
        print("2. Update CUDA toolkit to match driver version (12.9)")
        print("3. Reinstall PyTorch with CUDA 12.9 support")
        print("4. Check for conflicting CUDA processes")

if __name__ == "__main__":
    main()

