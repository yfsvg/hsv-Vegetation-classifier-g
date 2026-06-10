import sys
import os
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

MODEL_PATH = "best.pt" 
IMAGE_PATH = "bare.png" 

def predict(image_path: str, model_path: str):
    model = YOLO(model_path)
    results = model.predict(source=image_path, verbose=False)

    for result in results:
        probs = result.probs
        class_names = result.names

        top1_idx = probs.top1      
        # receive ocnfidence and label      
        top1_conf = probs.top1conf.item()
        top1_name = class_names[top1_idx]

        print(f"Overall prediction: {top1_name.upper()}")
        print(f"Confidence / confusion: {top1_conf:.1%}")
        # current busy working on this

        # Show all class probabilities ranked in confidence
        print("TOTAL confidences for each class - ")
        all_probs = probs.data.tolist()
        ranked = sorted(
            [(class_names[i], all_probs[i]) for i in range(len(all_probs))],
            key=lambda x: x[1],
            reverse=True
        )
        for label, prob in ranked:
            print(f" {label} {prob:.1%} ")


if __name__ == "__main__":
    # Allow overriding paths via command line args

    img = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    model = sys.argv[2] if len(sys.argv) > 2 else MODEL_PATH

    predict(img, model)