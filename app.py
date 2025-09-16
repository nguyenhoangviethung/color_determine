import cv2
import numpy as np
from ultralytics import YOLO
from color_utils import get_top_k_colors
from image_processing import preprocess_image, get_instance_segments, draw_segments_only, draw_segments_and_colors
import gradio as gr

def process_image(image, scale_factor=1.5, k=5):
    if image is None:
        return None, None, "Error: No image provided."
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    model = YOLO("deepfashion2_yolov8s-seg.pt")

    try:
        processed_image = preprocess_image(image, scale_factor)
    except ValueError as e:
        return None, None, f"Error: {e}"

    masks = get_instance_segments(processed_image, model)
    if not masks:
        return None, None, "No segments detected."

    top_colors_list = [get_top_k_colors(processed_image, m, k=k, bins=64) for m in masks]
    mask_images = draw_segments_only(processed_image, masks)
    annotated_image, color_text = draw_segments_and_colors(processed_image, masks, top_colors_list)

    return mask_images, annotated_image, color_text

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Color Detection and Segmentation App")
        gr.Markdown("Upload an image to detect clothing segments and their dominant colors.")
        submit_button = gr.Button("Process Image")
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Image")
            scale_factor = gr.Slider(minimum=1.0, maximum=3.0, value=1.5, step=0.1, label="Scale Factor")
            k_colors = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Top Colors")

        with gr.Row():
            mask_gallery = gr.Gallery(label="Segmented Masks")
            annotated_image = gr.Image(label="Annotated Image")
        color_output = gr.Textbox(label="Dominant Colors")
        submit_button.click(
            fn=process_image,
            inputs=[image_input, scale_factor, k_colors],
            outputs=[mask_gallery, annotated_image, color_output]
        )
    demo.launch(share=True)

if __name__ == "__main__":
    main()
