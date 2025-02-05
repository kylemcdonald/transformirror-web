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
    return parser.parse_args()

class WebcamTest:
    def __init__(self, camera_id=0):
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
            
            flipped_frame = self.current_frame.get_texture().get_transform(flip_y=True, flip_x=True)
            flipped_frame.anchor_x = 0
            flipped_frame.anchor_y = 0
            flipped_frame.blit(0, 0)#, width=flipped_frame.width, height=flipped_frame.height)
            
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
    app = WebcamTest(camera_id=args.camera_id)
    app.run()