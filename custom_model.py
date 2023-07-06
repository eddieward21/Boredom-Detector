import torch
import os 
from matplotlib import pyplot as plt
import numpy as np
import cv2


model = torch.hub.load('ultralytics/yolov5', 'custom', path = "yolov5/runs/train/exp7/weights/best.pt", force_reload=True)

img = os.path.join('data', 'images', 'bored.2e537528-18f6-11ee-9849-1e0033118c25.jpg')
result = model(img)
result.print()

plt.imshow(np.squeeze(result.render()))
plt.show()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)

    cv2.imshow("Boredom Detection", np.squeeze(results.render()))
    print(results.render())
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()