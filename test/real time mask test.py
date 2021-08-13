import cv2
from cvfwVer2 import CVFW_MODEL
import numpy as np


model = CVFW_MODEL(dsize=(64, 64))

model.add_directory(class_name="with mask", path="C:/kimdonghwan/python/CVFW/image/mask/train/with_mask")
model.add_directory(class_name="without mask", path="C:/kimdonghwan/python/CVFW/image/mask/train/without_mask")

model.train()

with_mask_modeling = cv2.resize(model.modeling(class_name="with mask"), dsize=(300, 300))
with_mask_modeling *= 255 / np.max(with_mask_modeling)

webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

count = 0

with_mask = False

mask_gray = []

while webcam:
    ret, img = webcam.read()

    gray = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1)
    # img = cv2.flip(img ,1)


    color = (0, 0, 0)
    if with_mask: color = (255, 255, 255)
    elif not with_mask: color = (0, 0, 0)
    cv2.rectangle(gray, (165, 75), (475, 385), color, 2)
    mask_gray = gray[80:380, 170:470]

    if count % 3 == 0:
        try:
            face = cv2.resize(mask_gray, dsize=(64, 64))
            predict = model.modeling_predict_class(face.flatten().tolist())
            with_mask = predict.index(min(predict)) == 0
            print(with_mask)

        except: pass

    count += 1

    alpha = 0.75
    blended = (gray[80:380, 170:470] * alpha) + (with_mask_modeling * (1 - alpha)).astype(np.uint8)
    gray[80:380, 170:470] = blended

    cv2.imshow('video', gray)
    k = cv2.waitKey(1) & 0xff
    if k == 27: break