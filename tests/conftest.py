import pytest
from surya.model.ocr_error.model import load_model as load_ocr_error_model, load_tokenizer as load_ocr_error_processor

@pytest.fixture(scope="session")
def ocr_error_model():
    ocr_error_m = load_ocr_error_model()
    ocr_error_p = load_ocr_error_processor()
    ocr_error_m.processor = ocr_error_p
    yield ocr_error_m
    del ocr_error_m