import argparse
import subprocess
import os


def run_app():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    ocr_app_path = os.path.join(cur_dir, "ocr_app.py")
    cmd = ["streamlit", "run", ocr_app_path]
    subprocess.run(cmd, env={**os.environ, "IN_STREAMLIT": "true"})

if __name__ == "__main__":
    run_app()