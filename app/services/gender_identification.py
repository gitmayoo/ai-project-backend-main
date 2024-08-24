import os
import cv2
from flask import jsonify
import torch

model_path = os.path.join(os.path.dirname(__file__), '..', "models","bestModel.pt")

camModel = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

def gender_detection(imagePath):
    image = cv2.imread(imagePath)
    frameRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = camModel(frameRGB)
    labels = results.names
    detected_classes = [labels[int(box[5])] for box in results.xyxy[0]]
    
    # Assume the first detected class is the relevant one
    if detected_classes:
        detected_class = detected_classes[0]
        return detected_class
    else:
        return 'No face detected'