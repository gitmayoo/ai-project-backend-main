import math
import cv2


# Image's desired dimension
DESIRED_HEIGHT = 256
DESIRED_WIDTH = 256

    
def image_resizer(image):
    # To get the width and size of the image
    h, w = image.shape[:2]
        # Checking the image's orientation (Landscape or Portrait)
    if h < w:
        new_img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        new_img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

    return new_img