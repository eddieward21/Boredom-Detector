import numpy as np
import cv2 
from matplotlib import pyplot as plt
import torch
import uuid 
import os
import time
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('videos/traffic.gif')
while cap.isOpened():

    ret, frame = cap.read()
    print(frame)
    results = model(frame)

    cv2.imshow("YOLO", np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

