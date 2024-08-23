import cv2
from flask import jsonify
import torch

camModel = torch.hub.load('ultralytics/yolov5', 'custom', path='bestModel.pt', force_reload=True)

def gender_detection():
    image = cv2.imread("uploads/53a013b7b03234d99cb20cf346f77b88.jpg")
    frameRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = camModel(frameRGB)
    labels = results.names
    detected_classes = [labels[int(box[5])] for box in results.xyxy[0]]
    
    # Assume the first detected class is the relevant one
    if detected_classes:
        detected_class = detected_classes[0]
        return jsonify({'gender': detected_class})
    else:
        return jsonify({'error': 'No face detected'})