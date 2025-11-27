#!/usr/bin/env python3
"""
Wrapper to run video inference and print JSON result for Node to parse:
Usage:
  python inference_video.py /path/to/video.mp4
"""
import sys, json
from model import load_model, predict_video

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"label":"ERROR","confidence":0,"error":"No path"}))
        return
    p = sys.argv[1]
    model = load_model()
    res = predict_video(model, p, num_samples=6)
    print(json.dumps(res))

if __name__ == "__main__":
    main()
