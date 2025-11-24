import onnxruntime as ort
import numpy as np
from PIL import Image
import math
import os

MODEL_PATH = "model_cardboard.onnx"
IMG_SIZE = 224

IMAGES = [
    "sampletest10.jpg",
    "sampletest11.jpg",
    "randtest.jpg",
    "testsample.jpg",
    "sample1.jpg",
    "testtest.jpg"
]

CLASS_NAMES = ["defective", "undefective"]


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Loaded RGB model:", MODEL_PATH)
print("-" * 50)

def load_image(path):

    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    arr = np.array(img).astype(np.float32) / 255.0  

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    arr = (arr - mean) / std

    arr = np.transpose(arr, (2, 0, 1))

    arr = np.expand_dims(arr, axis=0).astype(np.float32)

    return arr

def run_image(path):
    print("Testing:", path)

    inp = load_image(path)
    inp = inp.astype(np.float32)  

    raw = session.run([output_name], {input_name: inp})[0]
    logits = raw[0]  

    probs = softmax(logits)
    cls = int(np.argmax(probs))

    print(f"Prediction: {CLASS_NAMES[cls].upper()}  |  score={probs[cls]:.4f}")
    print("-" * 50)

for img in IMAGES:
    if not os.path.exists(img):
        print(f"File missing: {img}")
    else:
        run_image(img)
