import gradio as gr
from surya.detection import batch_detection
from surya.model.detection.segformer import load_model, load_processor
from surya.postprocessing.heatmap import draw_polys_on_image

model, processor = load_model(), load_processor()

HEADER = """
# Surya OCR Demo

This demo will let you try surya, a multilingual OCR model.  It supports text detection now, but will support text recognition in the future.

Notes:
- This works best on documents with printed text.
- Set DETECTOR_MODEL_CHECKPOINT=vikp/line_detector_math before running this app if you want better math detection.

Learn more [here](https://github.com/VikParuchuri/surya).
""".strip()

def text_detection(img):
    preds = batch_detection([img], model, processor)[0]
    img = draw_polys_on_image(preds["polygons"], img)
    return img, preds


with gr.Blocks() as app:
    gr.Markdown(HEADER)
    with gr.Row():
        input_image = gr.Image(label="Input Image", type="pil")
        output_image = gr.Image(label="Output Image", type="pil", interactive=False)
    text_detection_btn = gr.Button("Run Text Detection")

    json_output = gr.JSON(label="JSON Output")
    text_detection_btn.click(fn=text_detection, inputs=input_image, outputs=[output_image, json_output], api_name="text_detection")


if __name__ == "__main__":
    app.launch()
