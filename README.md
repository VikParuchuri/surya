# Surya

Surya is a multilingual document OCR toolkit that can currently do text line detection, but will soon have more functions.  It can run on CPU or GPU.

Surya is named after the [Hindu sun god](https://en.wikipedia.org/wiki/Surya), who has universal vision.

## Community

[Discord](https://discord.gg//KuZwXNGnfH) is where we discuss future development.

# Installation

You'll need python 3.9+ and PyTorch. You may need to install the CPU version of torch first if you're not using a Mac or a GPU machine.  See [here](https://pytorch.org/get-started/locally/) for more details.

Install with:

```
`pip install surya-ocr`
```

Model weights will automatically download the first time you run each function.

# Usage

- Inspect the settings in `surya/settings.py`.  You can override any settings with environment variables.
- Your torch device will be automatically detected, but you can override this.  For example, `TORCH_DEVICE=cuda`.  Note that the `mps` device has a bug (on the [Apple side](https://github.com/pytorch/pytorch/issues/84936)) that may prevent it from working properly.

## Text line detection

## Text recognition

Coming soon.

## Table and chart detection

Coming soon.

# Manual install

If you want to develop surya, you can install it manually:

- `git clone https://github.com/VikParuchuri/surya.git`
- `cd surya`
- `poetry install` # Installs main and dev dependencies

# Limitations


# Benchmarks


## Running your own benchmarks

You can benchmark the performance of surya on your machine.  

- Follow the manual install instructions above.
- `poetry install --group dev` # Installs dev dependencies
- Download the benchmark data [here](https://drive.google.com/file/d/1dbY0kBq2SUa885gmbLPUWSRzy5K7O5XJ/view?usp=sharing) and put it in the `data` folder.

*Text line detection*

```
python benchmark/detection.py
```

# Training



# Commercial usage



# Thanks

