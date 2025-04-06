import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model = YOLO("weights\yolov11n_320x320_40.pt")
source_path = "test_data\images"
for img_name in os.listdir(source_path):
    img = cv2.imread(os.path.join(source_path, img_name))
    predictions = model.predict(source=img, conf=0.1, verbose=False)[0]
    predictions = predictions.boxes.xywh
    if len(predictions) > 0:
        x, y, w, h = predictions[0]
        intbox = tuple(map(int, (x-w/2, y-h/2, x+w/2, y+h/2)))
        cv2.rectangle(img, intbox[0:2], intbox[2:4], color=(0, 255, 0), thickness=2)
    plt.imshow(img)
    plt.show()