import numpy as np
import cv2
from numpy.core.function_base import linspace
from os import listdir
import pickle

class CVIW:
    def __init__(self, m, n, P) -> None:
        self.m = m
        self.n = n
        self.weight = np.zeros([50000, 2])
        self.P = (np.array(P) + 1).tolist()

    
    def Weight_(self):
        for x in range(self.m-1):
            for y in range(self.n-1):
                # right
                w_ =  self.P[self.m * y + x + 1] / self.P[self.m * y + x]
                self.weight[self.m * y + x][0] = w_
                
                # down
                w_ = self.P[self.m * (y+1) + x] / self.P[self.m * y + x]
                self.weight[self.m * y + x][1] = w_

        for x in range(self.m-1):
            # right
            w_ = self.P[self.m * (self.n - 1) + x + 1] / self.P[self.m * (self.n-1) + x]
            self.weight[self.m * (self.n-1) + x][0] = w_

        for y in range(self.n-1):
            # down
            w_ = self.P[self.m * (y+2) - 1] / self.P[self.m * (y+1) - 1]
            self.weight[self.m * (y+1) - 1][1] = w_
	


# cost function class
class COST_FUNCTION:
    def __init__(self, weights = []) -> None:
        self.weights = sorted(weights)    # 가중치들
        self.avg_r = (weights[-1] - weights[0]) / (len(weights) - 1)    # 평균거리
        self.iw_count = 0
        self.single_group_c = 0  # 원소의 개수가 하나인 그룹
        self.iw = []  # 특징 그룹의 원소들의 평균값
        self.mean_groups = []  # 모든 그룹의 원소들의 평균값
        self.group_c = []  # 모든 그룹의 원소들의 개수


    # 가중치를 추가하는 메소드
    def add_weight(self, weight) -> None:
        self.weights = np.append(self.weights, weight)


    # 특징 그룹들의 범위를 골라내는 메소드
    def important_weight(self) -> None:

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


        self.single_group_c = self.group_c.count(1)    # 원소의 개수가 1개인 그룹의 개수
        if self.single_group_c == 0: self.single_group_c = 1

        for i in range(len(self.group_c)):
            if self.group_c[i] > self.single_group_c:    # 그룹 원소의 개수가 싱글그룹개수보다 많다면
                self.iw.append(self.mean_groups[i])
                self.iw_count = 1


    # cost method
    def cost(self, weight):
        if len(self.iw) == 0: return 0
        return np.min(abs(self.iw - weight))

        # np_group_range = np.array(self.group_range).flatten()
        # abs(np_group_range - weight)

        # for i in range(len(self.group_range)):
        #     start = self.group_range[i][0]
        #     end = self.group_range[i][1]

        #     if weight >= start and weight <= end: return 0

        # return 1


# CVIW GROUP class
class CVIW_GROUP:
    def __init__(self, class_name = "", dsize = (128, 128), cviws = []) -> None:
        self.cviws: list = cviws
        self.class_name: str = class_name
        self.important_weight: list = []
        # self.important_weight[i][j] = COST_FUNCTION(weights)
        self.dsize: tuple = dsize
        self.iw_count = 0  # 특징 가중치의 개수


    # add cviw method, img is flatten list of pixel data to add.
    def add_cviw(self, img) -> None:
        cviw = CVIW(self.dsize[0], self.dsize[1], img)  # CVIW 클래스 선언
        self.cviws.append(cviw)  # cviws 에 cviw 추가.


    # all cviw in cviws list get weight.
    def weight_all(self) -> None:
        for i in range(len(self.cviws)):
            self.cviws[i].Weight_()  # 모든 cviw 에 대하여 가중치를 구한다.

    
    # load then add cviw method.
    def load_add_cviw(self, file):
        img = cv2.resize(cv2.cvtColor(cv2.imread(file, 1), cv2.COLOR_BGR2GRAY), dsize=self.dsize).flatten().tolist()  # 로드한 이미지를 flatten 한다.
        cviw = CVIW(self.dsize[0], self.dsize[1], img)  # CVIW 클래스 선언
        self.cviws.append(cviw)  # cviws 에 해당 cviw 를 추가한다.

    
    # train method.
    def train_(self):
        self.weight_all()  # 가중치를 먼저 구함

        percent = self.dsize[0] * self.dsize[1]

        for i in range(self.dsize[0] * self.dsize[1]):
            self.important_weight.append([])
            for j in range(2):
                weights = []
                for cviw in self.cviws:
                    weights.append(cviw.weight[i][j])
                
                cost = COST_FUNCTION(sorted(weights))  # cost function 으로 특징가중치를 처리한다.
                cost.important_weight()  # 특징 그룹을 구하는 메소드.
                self.iw_count += cost.iw_count  # 특징가중치의 존재 여부(0, 1) 를 iw count 에 더한다.
                self.important_weight[i].append(cost)  # 특징가중치 리스트에 해당 cost function 클래스 대입
            
            print(f'{i / percent * 100} %', end="\r")

        print(self.iw_count)
        
        del self.cviws


    # cost 를 구하는 메소드.
    def cost_function(self, weight):
        cost = 0

        for i in range(self.dsize[0] * self.dsize[1]):
            for j in range(2):
                cost += self.important_weight[i][j].cost(weight[i][j])

        cost /= self.iw_count
        return cost



# CVIW 모델 클래스
class CVIW_MODEL:
    def __init__(self, dsize = (128, 128)) -> None:
        self.cviw_groups = []
        self.classes = []
        self.dsize = dsize

    
    # 디렉토리 경로로 CVIW_GROUP 을 추가하는 메소드.
    def add_directory(self, class_name = "", path = "") -> None:
        self.classes.append(class_name)
        imgs = listdir(path)
        cviw_group = CVIW_GROUP(class_name=class_name, dsize = self.dsize, cviws=[])

        for img in imgs:
            cviw_group.load_add_cviw(file = f'{path}\\{img}')

        self.cviw_groups.append(cviw_group)


    # 학습하는 메소드
    def train(self) -> None:
        for i in range(len(self.classes)):
            self.cviw_groups[i].train_()
            print(f"\n{self.classes[i]} Done!\n")


    # 클래스를 예측하는 메소드
    def predict_class(self, img):
        cviw = CVIW(self.dsize[0], self.dsize[1], img)
        cviw.Weight_()

        cost_list = []

        for cviw_group in self.cviw_groups:
            cost_list.append(cviw_group.cost_function(cviw.weight))
        
        print(cost_list)
        print(cost_list[0] / sum(cost_list), cost_list[1] / sum(cost_list))
        return self.classes[np.argmin(cost_list)]


    # 모델을 저장하는 메소드.
    def save_model(self, model_name = ""):
        with open(f"{model_name}.p", 'wb') as file:
            pickle.dump(self.classes, file)
            pickle.dump(self.dsize, file)

            important_weight = []
            for cviw_group in self.cviw_groups:
                important_weight.append([])

                for iw in cviw_group.important_weight:
                    important_weight[-1].append(iw)

            pickle.dump(important_weight, file)


    # 모델을 로드하는 메소드.
    def load_model(self, model_path = ""):
        with open(model_path, 'rb') as file:
            self.classes = pickle.load(file)
            self.dsize = pickle.load(file)
            important_weight = pickle.load(file)

            for i in range(len(important_weight)):
                
                cviw_group = CVIW_GROUP(class_name=self.classes[i], dsize = self.dsize)
                cviw_group.important_weight = important_weight[i]

                self.cviw_groups.append(cviw_group)



if __name__ == '__main__':
    # Time Test
    from time import time
    from random import randint
    import matplotlib.pyplot as plt

    cviw_group = CVIW_GROUP(class_name="test", dsize=(128, 128))
    for i in range(500):
        cviw_group.add_cviw([randint(1, 256) for _ in range(128 * 128)])

    start_time = time()
    cviw_group.train_()
    print(f"time: {time() - start_time}")


    # Cost Function
    cost_function = COST_FUNCTION(weights = [1, 7, 9, 14, 15, 23, 27])
    cost_function.important_weight()
    x = linspace(0, 50, 1000)
    y = [cost_function.cost(i) for i in x]
    plt.plot(x, y)
    plt.show()