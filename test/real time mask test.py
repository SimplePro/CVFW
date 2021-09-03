#final real time mask test python file

import cv2
from cvfw import CVFW_MODEL
import numpy as np


model = CVFW_MODEL(dsize=(64, 64), feature_group_number=10, feature_weight_number=300)

model.add_directory(class_name="with mask", path="C:/kimdonghwan/python/CVFW/image/mask/train/with_mask")
model.add_directory(class_name="without mask", path="C:/kimdonghwan/python/CVFW/image/mask/train/without_mask")

model.train()


mask_case = cv2.resize(cv2.imread("./mask_case2.png"), dsize=(300, 300))

mask_case_idx = []

for y in range(300):
    for x in range(300):
        if mask_case[y][x][0] == 0:
            mask_case_idx.append((y, x))


webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

count = 0

with_mask = False

mask_gray = []

color = (0, 0, 255)

with_masks = []

while webcam:
    ret, img = webcam.read()

    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    if count % 25 == 0:
        print(with_masks[-5:])

        if sum(with_masks[-5:]) / 5 >= 0.7:
            with_mask = True
            color = (0, 255, 0)

        else:
            with_mask = False
            color = (0, 0, 255)

        print(100 * sum(with_masks[-5:]) / 5, with_mask)
        
    
    cv2.rectangle(img, (165, 75), (475, 385), color, 2)
    mask_gray = gray[80:380, 170:470]

    if count % 5 == 0:
        try:
            face = cv2.resize(mask_gray, dsize=(64, 64))
            predict = model.modeling_predict_class(face.flatten().tolist())
            with_masks.append(int(predict.index(min(predict)) == 0))

        except: pass

    count += 1

    for y, x in mask_case_idx:
        img[y+80][x+170] = 0
    
    cv2.imshow('video', img)
    k = cv2.waitKey(1) & 0xff
    if k == 27: break
