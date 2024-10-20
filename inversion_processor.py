import cv2

class InversionProcessor:
    def __init__(self):
        pass
    
    def __call__(self, img, prompt):
        return cv2.bitwise_not(img)