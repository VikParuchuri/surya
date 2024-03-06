import argparse
import subprocess
import os


def run_app():
    parser = argparse.ArgumentParser(description="Run the streamlit OCR app")
    parser.add_argument("--math", action="store_true", help="Use math model for detection", default=False)
    args = parser.parse_args()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    ocr_app_path = os.path.join(cur_dir, "ocr_app.py")
    cmd = ["streamlit", "run", ocr_app_path]
    if args.math:
        cmd.append("--")
        cmd.append("--math")
    subprocess.run(cmd)