## 얼굴인식을 YOLO 로 해봐야함.

import cv2
from cvfwVer2 import CVFW_MODEL

model = CVFW_MODEL(dsize=(64, 64))

model.add_directory(class_name="with mask", path="C:/kimdonghwan/python/CVFW/image/mask/train/with_mask")
model.add_directory(class_name="without mask", path="C:/kimdonghwan/python/CVFW/image/mask/train/without_mask")

model.train()


# 가중치 파일 경로
cascade_filename = './haarcascade_frontalface_alt.xml'
# 모델 불러오기
cascade = cv2.CascadeClassifier(cascade_filename)

webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

count = 0

while webcam:
    ret, img = webcam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors = 3, minSize = (20, 20))
    roi_gray = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x-50, y-50), (x + w + 50, y+h+50), (255, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        face = cv2.resize(roi_gray, dsize=(64, 64))
        predict = model.modeling_predict_class(face.flatten().tolist())
        print(model.classes[predict.index(min(predict))])

    count += 1



    cv2.imshow('video', img)
    k = cv2.waitKey(1) & 0xff
    if k == 27: break

webcam.release()
cv2.destroyAllWindows()
