import subprocess
import os


def texify_app_cli():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    ocr_app_path = os.path.join(cur_dir, "texify_app.py")
    cmd = ["streamlit", "run", ocr_app_path]
    subprocess.run(cmd, env={**os.environ, "IN_STREAMLIT": "true"})