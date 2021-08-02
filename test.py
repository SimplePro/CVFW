from cviwVer2 import CVIW_MODEL
import cv2
import os


cviw_model = CVIW_MODEL(dsize=(100, 100))

cviw_model.add_directory(class_name="female eyes", path="C:\\kimdonghwan\\python\\CVIW\\image\\train\\femaleeyes")
cviw_model.add_directory(class_name="male eyes", path="C:\\kimdonghwan\\python\\CVIW\\image\\train\\maleeyes")

cviw_model.train()

# cviw_model.save_model(model_name="test")    -> memory 가 너무 많이 소비됨.  메모리 효율을 많이 늘릴 방법을 찾아야 한다.

answer = 0
count = 0

for file in os.listdir("./image/test/femaleeyes"):
    female = cv2.resize(cv2.cvtColor(cv2.imread(f"C:\\kimdonghwan\\python\\CVIW\\image\\test\\femaleeyes\\{file}", 1), cv2.COLOR_BGR2GRAY), dsize=(100, 100)).flatten().tolist()
    predict = cviw_model.predict_class(female)
    print("label: female   ", "predict:", predict)
    if predict == "female eyes": answer += 1
    count += 1


print("--------------")

for file in os.listdir('./image/test/maleeyes'):
    male = cv2.resize(cv2.cvtColor(cv2.imread(f"C:\\kimdonghwan\\python\\CVIW\\image\\test\\maleeyes\\{file}", 1), cv2.COLOR_BGR2GRAY), dsize=(100, 100)).flatten().tolist()
    predict = cviw_model.predict_class(male)
    print("label: male   ", "predict:", predict)
    if predict == "male eyes": answer += 1
    count += 1

print(f"accuracy: {answer / count}")