import cv2
import numpy as np


def load_image(path, size=(256, 256)):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Impossible de charger : {path}")

    image = cv2.resize(image, size)
    return image


def quantize_image(image, levels=8):
    factor = 256 // levels
    quantized = image // factor
    return quantized.astype(np.uint8)
