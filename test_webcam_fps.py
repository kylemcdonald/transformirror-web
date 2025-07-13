import cv2
import time

CAMERA_INDEX = 0
WIDTH = 1920
HEIGHT = 1280
FPS = 15
NUM_FRAMES = 100
SKIP_FRAMES = 10

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

# Print actual settings
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Requested: {WIDTH}x{HEIGHT} @ {FPS} FPS")
print(f"Actual: {int(actual_width)}x{int(actual_height)} @ {actual_fps:.2f} FPS (reported by driver)")

# Skip initial frames
for i in range(SKIP_FRAMES):
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame during warmup")
        cap.release()
        exit(1)

# Measure incoming frame rate
frame_count = 0
start_time = time.time()
while frame_count < NUM_FRAMES:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame_count += 1
end_time = time.time()
duration = end_time - start_time
measured_fps = frame_count / duration if duration > 0 else 0
print(f"Captured {frame_count} frames in {duration:.2f} seconds")
print(f"Measured FPS: {measured_fps:.2f}")

cap.release() 