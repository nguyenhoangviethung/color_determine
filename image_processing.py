import cv2
import numpy as np
from PIL import Image

def preprocess_image(image, scale_factor=1.5):
    if image is None:
        raise ValueError("Ảnh đầu vào là None")
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(upscaled, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v_)
    hsv_clahe = cv2.merge([h_, s_, v_clahe])
    enhanced = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    return enhanced

def get_instance_segments(image, yoloseg_model):
    if image is None:
        raise ValueError("Ảnh đầu vào là None")
    results = yoloseg_model.predict(image, imgsz=640)
    masks = []
    img_h, img_w = image.shape[:2]
    for result in results:
        if result.masks is not None:
            for m in result.masks.data:
                mask = m.cpu().numpy().astype(np.uint8)
                if mask.shape != (img_h, img_w):
                    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0).astype(np.uint8) * 255
                masks.append(mask)
    if not masks:
        return []
    total_pixels = sum(np.sum(mask) for mask in masks)
    threshold = total_pixels / 6
    return sorted([m for m in masks if np.sum(m) >= threshold],
                  key=lambda x: np.sum(x), reverse=True)

def draw_segments_only(image, masks):
    results = []
    for mask in masks:
        img_copy = image.copy()
        overlay = img_copy.copy()
        overlay[mask > 0] = (0, 0, 255)
        cv2.addWeighted(overlay, 0.2, img_copy, 0.8, 0, img_copy)
        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        results.append(Image.fromarray(img_rgb))
    return results

def draw_segments_and_colors(image, masks, top_colors_list):
    img = image.copy()
    color_text = ""
    for idx, (mask, top_colors) in enumerate(zip(masks, top_colors_list)):
        if not top_colors:
            continue
        color_name = top_colors[0][0]
        if len(top_colors) >= 2 and top_colors[1][1] > (2/3) * top_colors[0][1]:
            color_name = f"{top_colors[0][0]}-{top_colors[1][0]}"
        overlay = img.copy()
        overlay[mask > 0] = (0, 150, 255)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

        if np.sum(mask) > 0:
            y_coords, x_coords = np.where(mask > 0)
            y_min, x_min = y_coords.min(), x_coords.min()
            cv2.putText(img, color_name, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4)
            cv2.putText(img, color_name, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        color_text += f"Instance {idx} top colors:\n"
        for cname, count in top_colors:
            color_text += f"  - {cname}: {count} pixels\n"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), color_text
