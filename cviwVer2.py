from ipywidgets.widgets.widget_string import Label
import numpy as np
import cv2
from numpy.core.function_base import linspace
from os import listdir
import pickle
from tqdm import tqdm
from multiprocess import Pool


# Weight method
def Weight(m, n, P):
    weight = np.zeros([50000, 2])
    P = (np.array(P) + 1).tolist()


    for x in range(m-1):
            for y in range(n-1):
                # right
                w_ =  P[m * y + x + 1] / P[m * y + x]
                weight[m * y + x][0] = w_
                
                # down
                w_ = P[m * (y+1) + x] / P[m * y + x]
                weight[m * y + x][1] = w_


    for x in range(m-1):
        # right
        w_ = P[m * (n - 1) + x + 1] / P[m * (n-1) + x]
        weight[m * (n-1) + x][0] = w_


    for y in range(n-1):
        # down
        w_ = P[m * (y+2) - 1] / P[m * (y+1) - 1]
        weight[m * (y+1) - 1][1] = w_


    return weight



# cost function class
class COST_FUNCTION:
    def __init__(self, weights = [], feature_group_number = 3, feature_weight_number = 50) -> None:
        self.weights = sorted(weights)    # 가중치들
        self.avg_r = (weights[-1] - weights[0]) / (len(weights) - 1)    # 평균거리
        self.iw_count = 0
        self.iw = []  # 특징 그룹의 원소들의 평균값
        self.mean_groups = []  # 모든 그룹의 원소들의 평균값
        self.group_c = []  # 모든 그룹의 원소들의 개수
        self.feature_group_number = feature_group_number
        self.feature_weight_number = feature_weight_number


    # 가중치를 추가하는 메소드
    def add_weight(self, weight) -> None:
        self.weights = np.append(self.weights, weight)


    # 특징 그룹들의 범위를 골라내는 메소드
    def feature_weight(self) -> None:

        count = 1
        S = self.weights[0]
        for i in range(len(self.weights)-1):
            if self.weights[i+1] - self.weights[i] <= self.avg_r:
                S += self.weights[i+1]
                count += 1

            else:
                self.mean_groups.append(S / count)
                self.group_c.append(count)
                S = self.weights[i+1]
                count = 1
    
        self.iw = []

        self.iw_count = 1
        if len(self.mean_groups) < self.feature_group_number:
            if len(self.mean_groups) == 0: self.iw_count = 0

            else:
                for i in range(len(self.group_c)):
                    if self.group_c[i] > self.feature_weight_number:
                        self.iw.append(self.mean_groups[i])

        else:
            argsort_ = np.argsort(self.group_c)
            for i in range(len(argsort_)):
                if argsort_[i] < self.feature_weight_number and self.group_c[i] > self.feature_weight_number:
                    self.iw.append(self.mean_groups[i])

        self.weights = []  # 메모리 비움


    # cost method
    def cost_function(self, weight):
        if len(self.iw) == 0: return 0
        return np.min((self.iw - weight)**2)

    
    # update method
    def update__(self, feature_group_number, feature_weight_number):
        self.iw = []

        self.iw_count = 1
        if len(self.mean_groups) < feature_group_number:
            if len(self.mean_groups) == 0: self.iw_count = 0

            else:
                for i in range(len(self.group_c)):
                    if self.group_c[i] > feature_weight_number:
                        self.iw.append(self.mean_groups[i])

        else:
            argsort_ = np.argsort(self.group_c)
            for i in range(len(argsort_)):
                if argsort_[i] < feature_weight_number and self.group_c[i] > feature_weight_number:
                    self.iw.append(self.mean_groups[i])



# CVIW GROUP class
class CVIW_GROUP:
    def __init__(self, class_name = "", dsize = (128, 128)) -> None:
        self.cviws_weight: list = []
        self.class_name: str = class_name
        self.feature_weight: list = []
        # self.feature_weight[i][j] = COST_FUNCTION(weights)
        self.dsize: tuple = dsize
        self.iw_count = 0  # 특징 가중치의 개수

    
    def add_cviw_weight(self, img) -> None:
        self.cviws_weight.append(Weight(self.dsize[0], self.dsize[1], img))

    
    # load then add cviw method.
    def load_add_cviw(self, file):
        img = cv2.resize(cv2.cvtColor(cv2.imread(file, 1), cv2.COLOR_BGR2GRAY), dsize=self.dsize).flatten().tolist()  # 로드한 이미지를 flatten 한다.
        self.cviws_weight.append(Weight(self.dsize[0], self.dsize[1], img))

    
    # train method.
    def train_(self, feature_group_number, feature_weight_number):
        for i in tqdm(range(self.dsize[0] * self.dsize[1]), desc=self.class_name, mininterval=1):
            self.feature_weight.append([])
            for j in range(2):
                weights = []
                for w in self.cviws_weight:
                    weights.append(w[i][j])
                
                cost = COST_FUNCTION(sorted(weights))  # cost function 으로 특징가중치를 처리한다.
                cost.feature_weight()  # 특징 그룹을 구하는 메소드.
                self.iw_count += cost.iw_count  # 특징가중치의 존재 여부(0, 1) 를 iw count 에 더한다.
                self.feature_weight[i].append(cost)  # 특징가중치 리스트에 해당 cost function 클래스 대입
        
        self.cviws_weight = []  # 메모리 비움.

    
    # update method
    def update_(self, feature_group_number, feature_weight_number):
        self.iw_count = 0
        for i in range(self.dsize[0] * self.dsize[1]):
            for j in range(2):
                self.feature_weight[i][j].update__(feature_group_number, feature_weight_number)
                self.iw_count += self.feature_weight[i][j].iw_count


    # update cost method
    # def update_cost(self, weight):
    #     for i in range(self.dsize[0] * self.dsize[1]):
    #         for j in range(2):
    #             pass
                

    # cost 를 구하는 메소드.
    def cost(self, weight):
        return sum([self.feature_weight[i][j].cost_function(weight[i][j]) for i in range(self.dsize[0] * self.dsize[1]) for j in range(2)]) / self.iw_count



# CVIW 모델 클래스
class CVIW_MODEL:
    def __init__(self, dsize = (128, 128), feature_group_number = [3], feature_weight_number = [50]) -> None:
        self.cviw_groups = []
        self.classes = []
        self.dsize = dsize
        self.feature_group_number = feature_group_number
        self.feature_weight_number = feature_weight_number

    
    # 디렉토리 경로로 CVIW_GROUP 을 추가하는 메소드.
    def add_directory(self, class_name = "", path = "") -> None:
        self.classes.append(class_name)
        imgs = listdir(path)
        cviw_group = CVIW_GROUP(class_name=class_name, dsize = self.dsize)

        for img in imgs:
            cviw_group.load_add_cviw(file = f'{path}\\{img}')

        self.cviw_groups.append(cviw_group)


    # 학습하는 메소드
    def train(self) -> None:
        for i in range(len(self.classes)):
            self.cviw_groups[i].train_(self.feature_group_number, self.feature_weight_number)


    # 클래스를 예측하는 메소드
    def predict_class(self, img):
        cviw_weight = Weight(self.dsize[0], self.dsize[1], img)

        cost_list = []

        for cviw_group in self.cviw_groups:
            cost_list.append(cviw_group.cost(cviw_weight))

        sum_cost_list = sum(cost_list)
        predict = []
        for i in cost_list:
            predict.append(i / sum_cost_list)

        return predict


    # 모델을 저장하는 메소드.
    def save_model(self, model_name = ""):
        with open(f"{model_name}.p", 'wb') as file:
            pickle.dump(self.classes, file)
            pickle.dump(self.dsize, file)

            feature_weight = []
            for cviw_group in self.cviw_groups:
                feature_weight.append([])

                for iw in cviw_group.feature_weight:
                    feature_weight[-1].append(iw)

            pickle.dump(feature_weight, file)


    # 모델을 로드하는 메소드.
    def load_model(self, model_path = ""):
        with open(model_path, 'rb') as file:
            self.classes = pickle.load(file)
            self.dsize = pickle.load(file)
            feature_weight = pickle.load(file)

            for i in range(len(feature_weight)):
                
                cviw_group = CVIW_GROUP(class_name=self.classes[i], dsize = self.dsize)
                cviw_group.feature_weight = feature_weight[i]

                self.cviw_groups.append(cviw_group)



# CVIW Update Class
class CVIW_UPDATE:
    def __init__(self, cviw_model, feature_group_number = [3], feature_weight_number = [50]) -> None:
        self.cviw_model = cviw_model
        self.feature_group_number = feature_group_number
        self.feature_weight_number = feature_weight_number
        self.validation_path = []
        self.label = []
        self.accuracy_list = []


    # add validation method
    def add_validation(self, class_name, path):
        self.label.append(class_name)
        self.validation_path.append(path)
        

    # setting method
    def set(self, feature_group_number, feature_weight_number):
        self.feature_group_number = feature_group_number
        self.feature_weight_number = feature_weight_number


    # update method
    def update(self):
        for fgn in self.feature_group_number:
            for fwn in self.feature_weight_number:
                for i in range(len(self.cviw_model.cviw_groups)):
                    self.cviw_model.cviw_groups[i].update_(feature_group_number = fgn, feature_weight_number = fwn)
                
                self.accuracy_list.append(self.accuracy())
                print(f"feature_group_number: {fgn}, feature_weight_nummber: {fwn} Done!, accuracy: {self.accuracy_list[-1]}")

                
    # accuracy method
    def accuracy(self):
        true = 0
        count = 0

        for i in range(len(self.validation_path)):
            for file in listdir(self.validation_path[i]):
                img = cv2.resize(cv2.cvtColor(cv2.imread(f"{self.validation_path[i]}\\{file}", 1), cv2.COLOR_BGR2GRAY), dsize=self.cviw_model.dsize).flatten().tolist()
                predict = self.cviw_model.predict_class(img)
                if predict.index(min(predict)) == i: true += 1
                count += 1

        return true / count


    # print result method
    def result(self):
        i = 0
        for fgn in self.feature_group_number:
            for fwn in self.feature_weight_number:
                print(f"feature_group_number: {fgn}, feature_weight_number: {fwn}, accuracy:{self.accuracy_list[i]}")
                i += 1


if __name__ == '__main__':
    # Time Test
    from time import time
    from random import randint
    import matplotlib.pyplot as plt

    cviw_group = CVIW_GROUP(class_name="test", dsize=(128, 128))
    for i in range(500):
        cviw_group.add_cviw_weight([randint(1, 256) for _ in range(128 * 128)])

    start_time = time()
    cviw_group.train_()
    print(f"time: {time() - start_time}")


    # Cost Function
    cost_function = COST_FUNCTION(weights = [1, 7, 9, 14, 15, 23, 27])
    cost_function.feature_weight()
    x = linspace(0, 50, 1000)
    y = [cost_function.cost_function(i) for i in x]
    plt.plot(x, y)
    plt.show()