import pytest
from surya.model.ocr_error.model import load_model as load_ocr_error_model, load_tokenizer as load_ocr_error_processor
from surya.model.layout.model import load_model as load_layout_model
from surya.model.layout.processor import load_processor as load_layout_processor

@pytest.fixture(scope="session")
def ocr_error_model():
    ocr_error_m = load_ocr_error_model()
    ocr_error_p = load_ocr_error_processor()
    ocr_error_m.processor = ocr_error_p
    yield ocr_error_m
    del ocr_error_m


@pytest.fixture(scope="session")
def layout_model():
    layout_m = load_layout_model()
    layout_p = load_layout_processor()
    layout_m.processor = layout_p
    yield layout_m
    del layout_m

