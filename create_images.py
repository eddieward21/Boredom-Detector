import uuid
import time
import cv2
import os
import numpy as np
IMAGES_PATH = os.path.join("data", "images")
labels = ['bored', 'engaged']
number_imgs = 20

cap = cv2.VideoCapture(0)
for label in labels:
    print(f"collecting images for {label}")
    time.sleep(5)
    for i in range(number_imgs):
        print(f"collecting images for {label}. image number {i}")
        ret, frame = cap.read()
        image_name = os.path.join(IMAGES_PATH, label+ "." + str(uuid.uuid1()) + ".jpg")
        cv2.imwrite(image_name, frame)
        cv2.imshow("Collecting Images", frame)
        time.sleep(1)
        print(image_name)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


