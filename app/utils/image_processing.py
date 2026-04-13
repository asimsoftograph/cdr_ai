from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def crop_field(image: Image.Image, xyxy: Tuple[float, float, float, float]) -> Image.Image:
    left, top, right, bottom = map(int, xyxy)
    return image.crop((left, top, right, bottom))


def apply_clahe(image: Image.Image) -> Image.Image:
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))


def prepare_for_recognition(image: Image.Image) -> Image.Image:
    return apply_clahe(image)
