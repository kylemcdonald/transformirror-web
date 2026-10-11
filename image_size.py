import os


def image_size():
    """(width, height) the GPU workers run at, from TRANSFORMIRROR_IMAGE_SIZE:
    a single number for a square ("1080") or WIDTHxHEIGHT ("1280x720")."""
    value = os.getenv("TRANSFORMIRROR_IMAGE_SIZE", "1080").strip().lower()
    width, _, height = value.partition("x")
    return int(width), int(height or width)
