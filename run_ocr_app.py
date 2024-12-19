import argparse
import subprocess
import os


# 定义 run_app 函数
def run_app():
    # 运行 OCR 应用 Streamlit 程序 
    cur_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前目录
    ocr_app_path = os.path.join(cur_dir, "ocr_app.py") # OCR 应用路径
    cmd = ["streamlit", "run", ocr_app_path]
    subprocess.run(cmd, env={**os.environ, "IN_STREAMLIT": "true"})

# 程序入口
if __name__ == "__main__":
    run_app()