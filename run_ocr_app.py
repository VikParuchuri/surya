import subprocess
import os


def run_app():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    ocr_app_path = os.path.join(cur_dir, "ocr_app.py")
    subprocess.run(["streamlit", "run", ocr_app_path])