import argparse
import os

from surya.benchmark.bbox import get_pdf_lines
from surya.postprocessing.heatmap import draw_bboxes_on_image

from surya.input.processing import open_pdf, get_page_images
from surya.settings import settings


def main():
    parser = argparse.ArgumentParser(description="Draw pymupdf line bboxes on images.")
    parser.add_argument("pdf_path", type=str, help="Path to PDF to detect bboxes in.")
    parser.add_argument("--results_dir", type=str, help="Path to JSON file with OCR results.", default=os.path.join(settings.RESULT_DIR, "pymupdf"))
    args = parser.parse_args()

    doc = open_pdf(args.pdf_path)
    page_count = len(doc)
    page_indices = list(range(page_count))

    images = get_page_images(doc, page_indices)
    doc.close()

    image_sizes = [img.size for img in images]
    pdf_lines = get_pdf_lines(args.pdf_path, image_sizes)

    folder_name = os.path.basename(args.pdf_path).split(".")[0]
    result_path = os.path.join(args.results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    for idx, (img, bboxes) in enumerate(zip(images, pdf_lines)):
        bbox_image = draw_bboxes_on_image(bboxes, img)
        bbox_image.save(os.path.join(result_path, f"{idx}_bbox.png"))

    print(f"Wrote results to {result_path}")

if __name__ == "__main__":
    main()

