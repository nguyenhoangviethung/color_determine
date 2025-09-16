import math
import numpy as np
from collections import defaultdict

COLOR_MAP = [
    ((0, 10, 50, 255, 50, 255), "Red"),
    ((11, 25, 50, 255, 50, 255), "Orange"),
    ((26, 35, 50, 255, 50, 255), "Yellow"),
    ((36, 85, 50, 255, 50, 255), "Green"),
    ((86, 100, 50, 255, 100, 255), "Light Blue"),
    ((101, 110, 50, 255, 50, 255), "Blue"),
    ((111, 140, 50, 255, 50, 255), "Purple"),
    ((141, 165, 50, 255, 50, 255), "Magenta"),
    ((166, 179, 50, 255, 50, 255), "Red"),
    ((0, 179, 0, 50, 200, 255), "White"),
    ((0, 179, 0, 50, 100, 199), "Gray"),
    ((0, 179, 0, 50, 0, 100), "Black"),
    ((86, 110, 50, 255, 30, 49), "Dark Blue"),
    ((111, 140, 50, 255, 30, 49), "Dark Purple")
]

def hsv_to_color_name(hsv_color):
    h, s, v = hsv_color
    for (h_min, h_max, s_min, s_max, v_min, v_max), color_name in COLOR_MAP:
        if h_min <= h <= h_max and s_min <= s <= s_max and v_min <= v <= v_max:
            return color_name

    min_distance = float('inf')
    closest_color = "Unknown"
    for (h_min, h_max, s_min, s_max, v_min, v_max), color_name in COLOR_MAP:
        h_mid = (h_min + h_max) / 2
        s_mid = (s_min + s_max) / 2
        v_mid = (v_min + v_max) / 2
        h_diff = min(abs(h - h_mid), 180 - abs(h - h_mid))
        distance = math.sqrt((h_diff * 0.5)**2 + (s - s_mid)**2 + (v - v_mid)**2)
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
    return closest_color

def get_top_k_colors(image, mask, k=3, bins=64):
    import cv2
    if image is None or mask is None:
        return []
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masked_pixels = hsv_image[mask > 0]
    if len(masked_pixels) < 50:
        return []

    h_quantized = (masked_pixels[:, 0] // (180 // bins)) * (180 // bins)
    quantized = np.column_stack((h_quantized, masked_pixels[:, 1], masked_pixels[:, 2]))

    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    color_count_map = defaultdict(int)

    for i, color in enumerate(colors):
        color_name = hsv_to_color_name(color.tolist())
        color_count_map[color_name] += counts[i]
    sorted_colors = sorted(color_count_map.items(), key=lambda x: x[1], reverse=True)
    return sorted_colors[:k]
