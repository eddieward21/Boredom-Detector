import torch
model = torch.hub.load('ultralytics/yolov5', 'custom', path = "yolov5/runs/train/exp4/weights/last.pt", force_reload=True)
import os 
from matplotlib import pyplot as plt
import numpy as np

img = os.path.join('data', 'images', 'bored.00a34d96-16ef-11ee-a1e8-1e0033118c25.jpg')
result = model(img)
result.print()

plt.imshow(np.squeeze(result.render()))
plt.show()