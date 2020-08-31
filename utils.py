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

def calculate_luminance_ratio(img, mask, height, width):
    reference_plate = np.float32(np.where(mask == 1, 1, 0))
    road_marking = np.float32(np.where(mask == 2, 1, 0))

    img = img.reshape(1, height, width)
    reference_plate = reference_plate.reshape(1, height, width)
    road_marking = road_marking.reshape(1, height, width)

    rp_mean = np.mean(np.multiply(img, reference_plate).reshape(-1))
    rm_mean = np.mean(np.multiply(img, road_marking).reshape(-1))

    ratio = rm_mean / (rp_mean + 0.01)
    if ratio > 2.0:
        ratio = 2.0;

    return ratio

