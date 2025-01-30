import cv2
import time

class InversionProcessor:
    def __init__(self):
        pass
    
    def __call__(self, img, prompt):
        inverted = cv2.bitwise_not(img)
        time.sleep(0.080)
        return inverted
