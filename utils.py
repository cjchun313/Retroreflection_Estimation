import numpy as np
import cv2

def generate_img_from_mask(mask, height, width):
    b = np.float32(np.where(mask == 2, 255, 0))
    g = np.float32(np.where(mask == 1, 255, 0))
    r = np.float32(np.zeros([height, width, 1]))
    res = cv2.merge((b, g, r))

    return res

def write_image(filepath, img):
    cv2.imwrite(filepath, img)
