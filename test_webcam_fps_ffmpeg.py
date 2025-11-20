import subprocess
import numpy as np
import time
import sys
import signal

WIDTH = 1920
HEIGHT = 1080
FPS = 30
NUM_FRAMES = 100
SKIP_FRAMES = 10

# Global variable to store the pipe for clean shutdown
pipe = None

# FFmpeg command to capture from webcam
ffmpeg = (
    "ffmpeg -hide_banner -loglevel error "
    f"-f v4l2 -input_format mjpeg -framerate {FPS} "
    f"-video_size {WIDTH}x{HEIGHT} -i /dev/video0 "
    "-f rawvideo -pix_fmt bgr24 -"
)

print(f"Starting FFmpeg capture: {WIDTH}x{HEIGHT} @ {FPS} FPS")
print(f"FFmpeg command: {ffmpeg}")

def cleanup_pipe():
    """Clean shutdown of ffmpeg process"""
    global pipe
    if pipe:
        print("\nCleaning up ffmpeg process...")
        # Send SIGTERM first
        pipe.terminate()
        try:
            pipe.wait(timeout=1)
        except subprocess.TimeoutExpired:
            # If SIGTERM doesn't work, force kill
            pipe.kill()
            pipe.wait()
        finally:
            pipe.stdout.close()

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print(f"\nReceived signal {signum}, shutting down...")
    cleanup_pipe()
    sys.exit(0)

# Set up signal handlers for clean shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    pipe = subprocess.Popen(ffmpeg.split(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
except FileNotFoundError:
    print("Error: ffmpeg not found. Please install ffmpeg.")
    sys.exit(1)

# Skip initial frames
print(f"Skipping {SKIP_FRAMES} warmup frames...")
for i in range(SKIP_FRAMES):
    raw = pipe.stdout.read(WIDTH * HEIGHT * 3)
    if not raw:
        print("Failed to grab frame during warmup")
        pipe.terminate()
        sys.exit(1)

# Measure incoming frame rate
frame_count = 0
start_time = time.time()
frame_times = []

print(f"Capturing {NUM_FRAMES} frames...")
while frame_count < NUM_FRAMES:
    frame_start = time.time()
    raw = pipe.stdout.read(WIDTH * HEIGHT * 3)
    frame_end = time.time()
    
    if not raw:
        print("Failed to grab frame")
        break
    
    frame = np.frombuffer(raw, np.uint8).reshape(HEIGHT, WIDTH, 3)
    frame_count += 1
    frame_times.append(frame_end - frame_start)
    
    # Print progress every 10 frames
    if frame_count % 10 == 0:
        elapsed = frame_end - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"Frame {frame_count}: Current FPS = {current_fps:.2f}")

end_time = time.time()
duration = end_time - start_time
measured_fps = frame_count / duration if duration > 0 else 0

print(f"\nResults:")
print(f"Captured {frame_count} frames in {duration:.2f} seconds")
print(f"Measured FPS: {measured_fps:.2f}")
print(f"Average frame time: {sum(frame_times)/len(frame_times)*1000:.2f} ms")
print(f"Min frame time: {min(frame_times)*1000:.2f} ms")
print(f"Max frame time: {max(frame_times)*1000:.2f} ms")

# Clean shutdown
cleanup_pipe() 