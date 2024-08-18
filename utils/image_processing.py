import cv2
import numpy as np

def apply_image_adjustments(image, brightness, contrast, saturation, hue, sharpness):
    # Convert to float32 for calculations
    img_float = image.astype(np.float32) / 255.0

    # Brightness
    brightness = brightness / 100.0
    img_float = np.clip(img_float + brightness, 0, 1)

    # Contrast
    contrast = contrast / 100.0
    img_float = np.clip((img_float - 0.5) * (1 + contrast) + 0.5, 0, 1)

    # Convert to HSV for saturation and hue adjustments
    img_hsv = cv2.cvtColor(img_float, cv2.COLOR_RGB2HSV)

    # Saturation
    saturation = saturation / 100.0
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * (1 + saturation), 0, 1)

    # Hue
    hue_shift = hue
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_shift / 2) % 180

    # Convert back to RGB
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    # Sharpness
    sharpness = sharpness / 10.0
    if sharpness > 0:
        blurred = cv2.GaussianBlur(img_rgb, (0, 0), 3)
        img_rgb = cv2.addWeighted(img_rgb, 1 + sharpness, blurred, -sharpness, 0)

    # Convert back to uint8
    return (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)