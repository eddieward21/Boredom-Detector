import numpy as np
import cv2 
from matplotlib import pyplot as plt
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

image = "videos/carla.jpeg"

result = model(image)
print(result.render())
plt.imshow(np.squeeze(result.render()))
plt.show()

