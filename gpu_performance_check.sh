#!/bin/bash

echo "=== GPU Performance Diagnostic Tool ==="
echo "Run this script on both systems to compare performance factors"
echo ""

echo "=== System Information ==="
echo "Hostname: $(hostname)"
echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"
echo ""

echo "=== CPU Information ==="
echo "CPU Model: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "CPU Cores: $(nproc)"
echo "CPU Frequency: $(lscpu | grep 'CPU max MHz' | cut -d: -f2 | xargs) MHz"
echo ""

echo "=== Memory Information ==="
free -h
echo ""

echo "=== GPU Information ==="
echo "GPU Model: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "Driver Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits)"
echo "CUDA Version: $(nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits)"
echo ""

echo "=== Current GPU Performance State ==="
nvidia-smi --query-gpu=performance_state,clocks.current.graphics,clocks.current.memory,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv
echo ""

echo "=== GPU Power and Temperature ==="
nvidia-smi -q -d POWER,TEMPERATURE | grep -E "(Current Temp|Power Draw|Power Limit)"
echo ""

echo "=== GPU Memory Usage ==="
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
echo ""

echo "=== Supported Clock Speeds ==="
echo "Max Graphics Clock: $(nvidia-smi -q -d CLOCK | grep 'Max Clocks' -A 5 | grep 'Graphics' | head -1 | awk '{print $3}') MHz"
echo "Max Memory Clock: $(nvidia-smi -q -d CLOCK | grep 'Max Clocks' -A 5 | grep 'Memory' | head -1 | awk '{print $3}') MHz"
echo ""

echo "=== Process Information ==="
echo "GPU Processes:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo ""

echo "=== System Load ==="
uptime
echo ""

echo "=== Disk I/O ==="
iostat -x 1 1 | tail -n +4
echo ""

echo "=== Network Interface ==="
ip link show | grep -E "state UP" -A 1
echo ""

echo "=== Python/CUDA Environment ==="
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
if python3 -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0)}')" 2>/dev/null; then
    python3 -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0)}')"
fi
echo ""

echo "=== Benchmark Test ==="
echo "Running quick GPU benchmark..."
python3 -c "
import torch
import time

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Testing on: {torch.cuda.get_device_name(0)}')
    
    # Warm up
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    for _ in range(10):
        z = torch.mm(x, y)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(100):
        z = torch.mm(x, y)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f'Matrix multiplication benchmark: {(end_time - start_time)*1000:.2f} ms for 100 iterations')
    print(f'Average per iteration: {((end_time - start_time)*1000)/100:.2f} ms')
else:
    print('CUDA not available')
"
echo ""

echo "=== Performance Recommendations ==="
echo "1. Check if both systems have the same GPU model and driver version"
echo "2. Compare current clock speeds vs maximum supported speeds"
echo "3. Check for thermal throttling (temperature > 80°C)"
echo "4. Verify power limits are set to maximum"
echo "5. Check for background processes using GPU"
echo "6. Compare CPU frequency scaling"
echo "7. Verify memory bandwidth and latency"
echo "8. Check for different CUDA/PyTorch versions"
echo "" 