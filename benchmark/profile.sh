#!/bin/bash
python -m cProfile -s time -o data/profile.pstats detect_text.py data/benchmark/nyt2.pdf --max 10
