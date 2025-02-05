import cv2
import pyglet
import numpy as np
import argparse

CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
CIRCLE_RADIUS = 50  # 20px diameter = 10px radius
CIRCLE_X = 941.0  # Fixed X position
CIRCLE_Y = 215.0  # Fixed Y position

def parse_args():
    parser = argparse.ArgumentParser(description='Webcam test with configurable camera ID')
    parser.add_argument('--camera-id', type=int, default=0,
                      help='Camera device ID (default: 0)')
    parser.add_argument('--exposure', type=float, default=None,
                      help='Camera exposure value (0-100, default: use camera default)')
    parser.add_argument('--brightness', type=float, default=None,
                      help='Camera brightness value (0-100, default: use camera default)')
    return parser.parse_args()

class WebcamTest:
    def __init__(self, camera_id=0, exposure=None, brightness=None):
        self.window = pyglet.window.Window(fullscreen=True)
        self.window.event(self.on_draw)
        self.window.event(self.on_mouse_motion)
        
        self.mouse_x = 0
        self.mouse_y = 0
        
        self.cap = cv2.VideoCapture(camera_id)
        
        # Print all supported camera properties
        props = [attr for attr in dir(cv2) if attr.startswith('CAP_PROP_')]
        print("\nSupported camera properties:")
        for prop in props:
            try:
                prop_id = getattr(cv2, prop)
                value = self.cap.get(prop_id)
                print(f"{prop}: {value}")
            except:
                continue
        
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 24)
        
        # Set and verify exposure if provided
        if exposure is not None:
            print(f"\nTrying to set exposure to: {exposure}")
            
            # Try method 1: Standard auto-exposure disable
            success1 = self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Try value 1
            if not success1:
                success1 = self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # Try value 0
            print(f"Method 1 - Disabled auto-exposure (1/0): {success1}")
            
            # Scale the exposure value to the camera's range (0-10000)
            # Map input range (0-100) to camera range (0-10000)
            scaled_exposure = min(10000, max(0, exposure * 100))
            
            success = self.cap.set(cv2.CAP_PROP_EXPOSURE, scaled_exposure)
            actual = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            print(f"\nExposure setting attempt:")
            print(f"  Input exposure value: {exposure}")
            print(f"  Scaled to camera range: {scaled_exposure}")
            print(f"  Set success: {success}")
            print(f"  Actual exposure: {actual}")
            
            print(f"\nFinal camera state:")
            print(f"Auto exposure mode: {self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
            print(f"Current exposure: {self.cap.get(cv2.CAP_PROP_EXPOSURE)}")

        # Set and verify brightness if provided
        if brightness is not None:
            print(f"\nTrying to set brightness to: {brightness}")
            # Map input range (0-100) to camera range (0-255)
            scaled_brightness = min(255, max(0, brightness * 2.55))
            
            success = self.cap.set(cv2.CAP_PROP_BRIGHTNESS, scaled_brightness)
            actual = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
            print(f"\nBrightness setting attempt:")
            print(f"  Input brightness value: {brightness}")
            print(f"  Scaled to camera range: {scaled_brightness}")
            print(f"  Set success: {success}")
            print(f"  Actual brightness: {actual}")
        
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"\nCamera initialized with resolution: {actual_width}x{actual_height}")
        
        self.current_frame = None
        
        pyglet.clock.schedule_interval(self.update_frame, 1/30.0)
        
        self.window.event(self.on_key_press)

    def update_frame(self, dt):
        ret, frame = self.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            self.current_frame = pyglet.image.ImageData(
                CAPTURE_WIDTH, CAPTURE_HEIGHT,
                'RGB',
                rgb_frame.tobytes(),
                pitch=CAPTURE_WIDTH * 3
            )

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_x = x
        self.mouse_y = y
        print(f"Mouse position: x={x}, y={y}")

    def on_draw(self):
        self.window.clear()
        if self.current_frame:
            window_width = self.window.width
            window_height = self.window.height
            
            scale_x = window_width / CAPTURE_WIDTH
            scale_y = window_height / CAPTURE_HEIGHT
            scale = min(scale_x, scale_y)
            
            x = (window_width - CAPTURE_WIDTH * scale) / 2
            scaled_height = CAPTURE_HEIGHT * scale
            y = CAPTURE_HEIGHT
            
            flipped_frame = self.current_frame.get_texture().get_transform(flip_y=True)
            flipped_frame.blit(x, y, width=CAPTURE_WIDTH * scale, height=scaled_height)
            
            # Draw black circle at fixed position
            circle = pyglet.shapes.Circle(
                x=CIRCLE_X,
                y=CIRCLE_Y,
                radius=CIRCLE_RADIUS,
                color=(0, 0, 0)
            )
            circle.draw()

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            pyglet.app.exit()

    def run(self):
        pyglet.app.run()
        self.cap.release()

if __name__ == '__main__':
    args = parse_args()
    app = WebcamTest(camera_id=args.camera_id, exposure=args.exposure, brightness=args.brightness)
    app.run()