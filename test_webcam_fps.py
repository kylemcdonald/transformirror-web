import cv2
import time
import sys

CAMERA_INDEX = 0
WIDTH = 1280
HEIGHT = 720
FPS = 30
NUM_FRAMES = 100
SKIP_FRAMES = 10

# cam_uri = "v4l2:///dev/video0?input_format=mjpeg&framerate=30&video_size=1920x1080"
# cap = cv2.VideoCapture(cam_uri, cv2.CAP_FFMPEG)
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

# Print actual settings
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cap.get(cv2.CAP_PROP_FOURCC)
fourcc_str = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
print(f"Requested: {WIDTH}x{HEIGHT} @ {FPS} FPS")
print(f"Actual: {int(actual_width)}x{int(actual_height)} @ {actual_fps:.2f} FPS")
print(f"FourCC: {fourcc_str}")

# Skip initial frames
print(f"Skipping {SKIP_FRAMES} warmup frames...")
for i in range(SKIP_FRAMES):
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame during warmup")
        cap.release()
        exit(1)

# Measure incoming frame rate
frame_count = 0
start_time = time.time()
frame_times = []

print(f"Capturing {NUM_FRAMES} frames...")
while frame_count < NUM_FRAMES:
    frame_start = time.time()
    ret, frame = cap.read()
    frame_end = time.time()
    
    if not ret:
        print("Failed to grab frame")
        break
    
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

cap.release() 