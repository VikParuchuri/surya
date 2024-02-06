# Surya

Surya is a multilingual document OCR toolkit.  It can do:

- Accurate line-level text detection in any language
- Text recognition in 90+ languages
- Table and chart detection (coming soon)

It works on a range of documents (see [usage](#usage) and [benchmarks](#benchmarks) for more details).

![New York Times Article Example](static/images/excerpt.png)

Surya is named after the [Hindu sun god](https://en.wikipedia.org/wiki/Surya), who has universal vision.

## Community

[Discord](https://discord.gg//KuZwXNGnfH) is where we discuss future development.

## Examples

| Name             | Text Detection                      |
|------------------|-------------------------------------|
| New York Times   | [Image](static/images/nyt.png)      |
| Japanese         | [Image](static/images/japanese.png) |
| Chinese          | [Image](static/images/chinese.png)  |
| Hindi            | [Image](static/images/hindi.png)    |
| Presentation     | [Image](static/images/pres.png)     |
| Scientific Paper | [Image](static/images/paper.png)    |
| Scanned Document | [Image](static/images/scanned.png)  |
| Scanned Form | [Image](static/images/funsd.png)    |

# Installation

You'll need python 3.9+ and PyTorch. You may need to install the CPU version of torch first if you're not using a Mac or a GPU machine.  See [here](https://pytorch.org/get-started/locally/) for more details.

Install with:

```
pip install surya-ocr
```

Model weights will automatically download the first time you run surya.

# Usage

- Inspect the settings in `surya/settings.py`.  You can override any settings with environment variables.
- Your torch device will be automatically detected, but you can override this.  For example, `TORCH_DEVICE=cuda`. Note that the `mps` device has a bug (on the [Apple side](https://github.com/pytorch/pytorch/issues/84936)) that may prevent it from working properly.

## OCR (text recognition)

You can detect text lines in an image, pdf, or folder of images/pdfs with the following command.  This will write out a json file with the detected text and bboxes, and optionally save images of the reconstructed page.

```
surya_ocr DATA_PATH --images --lang hi,en
```

- `DATA_PATH` can be an image, pdf, or folder of images/pdfs
- `--lang` specifies the language(s) to use for OCR.  You can comma separate multiple languages. Use the language name or two-letter ISO code from [here](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes).  Surya supports the 90+ languages found in `surya/languages.py`.
- `--lang_file` if you want to use a different language for different PDFs/images, you can specify languages here.  The format is a JSON dict with the keys being filenames and the values as a list, like `{"file1.pdf": ["en", "hi"], "file2.pdf": ["en"]}`.
- `--images` will save images of the pages and detected text lines (optional)
- `--results_dir` specifies the directory to save results to instead of the default
- `--max` specifies the maximum number of pages to process if you don't want to process everything
- `--start_page` specifies the page number to start processing from

The `results.json` file will contain these keys for each page of the input document(s):

- `text_lines` - the detected text in each line
- `polys` - the polygons for each detected text line in (x1, y1), (x2, y2), (x3, y3), (x4, y4) format.  The points are in clockwise order from the top left.
- `bboxes` - the axis-aligned rectangles for each detected text line in (x1, y1, x2, y2) format.  (x1, y1) is the top left corner, and (x2, y2) is the bottom right corner.
- `language` - the languages specified for the page
- `name` - the name of the file
- `page_number` - the page number in the file

**Performance tips**

Setting the `RECOGNITION_BATCH_SIZE` env var properly will make a big difference when using a GPU.  Each batch item will use `40MB` of VRAM, so very high batch sizes are possible.  The default is a batch size `256`, which will use about 10GB of VRAM.

Depending on your CPU core count, `RECOGNITION_BATCH_SIZE` might make a difference there too - the default CPU batch size is `32`.


### From Python

You can also do OCR from code with:

```
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

image = Image.open(IMAGE_PATH)
langs = ["en"] # Replace with your languages

det_processor = load_det_processor()
det_model = load_det_model()

rec_model = load_rec_model()
rec_processor = load_rec_processor()

predictions = run_ocr([image], langs, det_model, det_processor, rec_model, rec_processor)
```


## Text line detection

You can detect text lines in an image, pdf, or folder of images/pdfs with the following command.  This will write out a json file with the detected bboxes, and optionally save images of the pages with the bboxes.

```
surya_detect DATA_PATH --images
```

- `DATA_PATH` can be an image, pdf, or folder of images/pdfs
- `--images` will save images of the pages and detected text lines (optional)
- `--max` specifies the maximum number of pages to process if you don't want to process everything
- `--results_dir` specifies the directory to save results to instead of the default

The `results.json` file will contain these keys for each page of the input document(s):

- `polygons` - polygons for each detected text line (these are more accurate than the bboxes) in (x1, y1), (x2, y2), (x3, y3), (x4, y4) format.  The points are in clockwise order from the top left.
- `bboxes` - axis-aligned rectangles for each detected text line in (x1, y1, x2, y2) format.  (x1, y1) is the top left corner, and (x2, y2) is the bottom right corner.
- `vertical_lines` - vertical lines detected in the document in (x1, y1, x2, y2) format.
- `horizontal_lines` - horizontal lines detected in the document in (x1, y1, x2, y2) format.
- `page_number` - the page number of the document

**Performance tips**

Setting the `DETECTOR_BATCH_SIZE` env var properly will make a big difference when using a GPU.  Each batch item will use `280MB` of VRAM, so very high batch sizes are possible.  The default is a batch size `32`, which will use about 9GB of VRAM.

Depending on your CPU core count, `DETECTOR_BATCH_SIZE` might make a difference there too - the default CPU batch size is `2`.

You can adjust `DETECTOR_NMS_THRESHOLD` and `DETECTOR_TEXT_THRESHOLD` if you don't get good results.  Try lowering them to detect more text, and vice versa.


### From Python

You can also do text detection from code with:

```
from PIL import Image
from surya.detection import batch_detection
from surya.model.segformer import load_model, load_processor

image = Image.open(IMAGE_PATH)
model, processor = load_model(), load_processor()

# predictions is a list of dicts, one per image
predictions = batch_detection([image], model, processor)
```

## Table and chart detection

Coming soon.

# Manual install

If you want to develop surya, you can install it manually:

- `git clone https://github.com/VikParuchuri/surya.git`
- `cd surya`
- `poetry install` # Installs main and dev dependencies

# Limitations

- Math will not be detected well with the main model.  Use `DETECTOR_MODEL_CHECKPOINT=vikp/line_detector_math` for better results.
- This is specialized for document OCR.  It will likely not work on photos or other images.
- It is for printed text, not handwriting.
- The model has trained itself to ignore advertisements.
- You can find language support for OCR in `surya/languages.py`.  Text detection should work with any language.

# Benchmarks

## OCR

Coming soon.

## Text line detection

![Benchmark chart](static/images/benchmark_chart_small.png)

| Model     |   Time (s) |   Time per page (s) |   precision |   recall |
|-----------|------------|---------------------|-------------|----------|
| surya     |    52.6892 |            0.205817 |    0.844426 | 0.937818 |
| tesseract |    74.4546 |            0.290838 |    0.631498 | 0.997694 |


Tesseract is CPU-based, and surya is CPU or GPU.  I ran the benchmarks on a system with an A6000 GPU, and a 32 core CPU.  This was the resource usage:

- tesseract - 32 CPU cores, or 8 workers using 4 cores each
- surya - 32 batch size, for 9GB VRAM usage

**Methodology**

Surya predicts line-level bboxes, while tesseract and others predict word-level or character-level.  It's also hard to find 100% correct datasets with line-level annotations. Merging bboxes can be noisy, so I chose not to use IoU as the metric for evaluation.

I instead used coverage, which calculates:

- Precision - how well predicted bboxes cover ground truth bboxes
- Recall - how well ground truth bboxes cover predicted bboxes

First calculate coverage for each bbox, then add a small penalty for double coverage, since we want the detection to have non-overlapping bboxes.  Anything with a coverage of 0.5 or higher is considered a match.

Then we calculate precision and recall for the whole dataset.

## Running your own benchmarks

You can benchmark the performance of surya on your machine.  

- Follow the manual install instructions above.
- `poetry install --group dev` # Installs dev dependencies

**Text line detection**

This will evaluate tesseract and surya for text line detection across a randomly sampled set of images from [doclaynet](https://huggingface.co/datasets/vikp/doclaynet_bench).

```
python benchmark/detection.py --max 256
```

- `--max` controls how many images to process for the benchmark
- `--debug` will render images and detected bboxes
- `--pdf_path` will let you specify a pdf to benchmark instead of the default data
- `--results_dir` will let you specify a directory to save results to instead of the default one


# Training

The text detection was trained on 4x A6000s for about 3 days.  It used a diverse set of images as training data.  It was trained from scratch using a modified segformer architecture that reduces inference RAM requirements.

Text recognition was trained on 4x A6000s for 2 weeks.  It was trained using a modified donut model (GQA, MoE layer, UTF-16 decoding, layer config changes).

# Commercial usage

The text detection and OCR models were trained from scratch, so they're okay for commercial usage.  The weights are licensed cc-by-nc-sa-4.0, but I will waive that for any organization under $5M USD in gross revenue in the most recent 12-month period.

If you want to remove the GPL license requirements for inference or use the weights commercially over the revenue limit, please contact me at surya@vikas.sh for dual licensing.

# Thanks

This work would not have been possible without amazing open source AI work:

- [Segformer](https://arxiv.org/pdf/2105.15203.pdf) from NVIDIA
- [Donut](https://github.com/clovaai/donut) from Naver
- [transformers](https://github.com/huggingface/transformers) from huggingface
- [CRAFT](https://github.com/clovaai/CRAFT-pytorch), a great scene text detection model

Thank you to everyone who makes open source AI possible.
