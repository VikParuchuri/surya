import gradio as gr
from PIL import Image, ImageDraw
from surya.detection import batch_inference
from surya.model.segformer import load_model, load_processor

model, processor = load_model(), load_processor()

def surya(img):

    # surya predictions is a list of dicts for the given image
    predictions = batch_inference([img], model, processor)
    bboxes = predictions[0]['bboxes']
    vertical_lines = predictions[0]['vertical_lines']
    horizontal_lines = predictions[0]['horizontal_lines']

    # Initialize the drawing context with the image as background
    draw = ImageDraw.Draw(img)

    # OCR predictions (replace the sample data with your actual OCR output)
    predictions = {
        'bboxes': bboxes, # bounding boxes data here
        'vertical_lines':  vertical_lines, # vertical lines data here
        'horizontal_lines':  horizontal_lines # your horizontal lines data here
    }

    # Draw bounding boxes
    for bbox in predictions['bboxes']:
        draw.rectangle(bbox, outline='red', width=2)

    # Draw vertical lines
    for vline in predictions['vertical_lines']:
        x1, y1, x2, y2 = vline['bbox']
        draw.line((x1, y1, x2, y2), fill='blue', width=2)

    # Draw horizontal lines
    for hline in predictions['horizontal_lines']:
        x1, y1, x2, y2 = hline['bbox']
        draw.line((x1, y1, x2, y2), fill='green', width=2)

    # return the final image 
    return img 

# Blocks API
with gr.Blocks() as demo:
  # title for the app
  gr.HTML("<h1><center> SURYA Demo </h1></center>")
  # input image component
  input_image = gr.Image(label="Input Image", type='pil')
  # run inference on the input image
  btn = gr.Button("Run Surya")
  # output image component
  output_image = gr.Image(label="Surya Output")
  btn.click(fn=surya, inputs=input_image, outputs=output_image, api_name="surya")


if __name__ == "__main__":
    demo.launch()