import cv2
from cvfwVer2 import CVFW_MODEL


model = CVFW_MODEL(dsize=(64, 64))

model.add_directory(class_name="with mask", path="C:/kimdonghwan/python/CVFW/image/mask/train/with_mask")
model.add_directory(class_name="without mask", path="C:/kimdonghwan/python/CVFW/image/mask/train/without_mask")

model.train()

webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

count = 0

with_mask = False

roi_gray = []

while webcam:
    ret, img = webcam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.flip(img ,1)


    color = (0, 0, 255)
    if with_mask: color = (0, 255, 0)
    elif not with_mask: color = (0, 0, 255)
    cv2.rectangle(img, (170, 80), (470, 380), color, 2)
    roi_gray = gray[80+20:380+20, 170:470]


    if count % 30 == 0:
        try:
            face = cv2.resize(roi_gray, dsize=(64, 64))
            predict = model.predict_class(face.flatten().tolist())
            with_mask = predict.index(min(predict)) == 0

        except: pass

    count += 1


    cv2.imshow('video', img)
    k = cv2.waitKey(1) & 0xff
    if k == 27: break