import numpy as np
import cv2
from numpy.core.function_base import linspace
from os import listdir
import pickle
from numba import cuda
# import code
# code.interact(local=locals)


class CVIW:
    def __init__(self, m, n, P) -> None:
        self.m = m
        self.n = n
        self.weight = np.zeros([50000, 8])
        self.P = (np.array(P) + 1).tolist()

    
    def Weight_(self):
        self.case1Weight()
        self.case2Weight()
        self.case3Weight()
        self.case4Weight()


    # first case weight method
    def case1Weight(self):
        m = self.m
        P = self.P

        for x in range(m - 1):
            for y in range(1, self.n - 1):
                # right up
                w_ = P[m * (y - 1) + x + 1] / P[m * y + x]
                self.weight[m * y + x][0] = w_
                self.weight[m * (y - 1) + x + 1][4] = 1 / w_

                # right
                w_ = P[m * y + x + 1] / P[m * y + x]
                self.weight[m * y + x][1] = w_
                self.weight[m * y + x + 1][5] = 1 / w_

                # right down
                w_ = P[m * (y + 1) + x + 1] / P[m * y + x]
                self.weight[m * y + x][2] = w_
                self.weight[m * (y + 1) + x + 1][6] = 1 / w_

                # down
                w_ = P[m * (y + 1) + x] / P[m * y + x]
                self.weight[m * y + x][3] = w_
                self.weight[m * (y + 1) + x][7] = 1 / w_


    # second case weight method
    def case2Weight(self):
        P = self.P
        m = self.m

        for x in range(m - 1):
            # right
            w_ = P[x + 1] / P[x]
            self.weight[x][1] = w_
            self.weight[x + 1][5] = 1 / w_

            # right down
            w_ = P[x + m + 1] / P[x]
            self.weight[x][2] = w_
            self.weight[x + m + 1][6] = 1 / w_

            # down
            w_ = P[x + m] / P[x]
            self.weight[x][3] = w_
            self.weight[x + m][7] = 1 / w_


    # third case weight method
    def case3Weight(self):
        y = self.n - 1
        m = self.m
        P = self.P

        for x in range(m - 1):
            # right up
            w_ = P[m * (y - 1) + x + 1] / P[m * y + x]
            self.weight[m * y + x][0] = w_
            self.weight[m * (y - 1) + x + 1][4] = 1 / w_

            # right
            w_ = P[m * y + x + 1] / P[m * y + x]
            self.weight[m * y + x][1] = w_
            self.weight[m * y + x + 1][5] = 1 / w_


    # fourth case weight method
    def case4Weight(self):
        P = self.P
        m = self.m
        x = m - 1

        for y in range(self.n - 1):
            # down
            w_ = P[m * (y + 1) + x] / P[m * y + x]
            self.weight[m * y + x][3] = w_
            self.weight[m * (y + 1) + x][7] = 1 / w_
	

    # print weight method
    def printWeight(self):
        for i in range(self.m * self.n):
            for j in range(8):
                print(round(self.weight[i][j], 3), end=" ")
            
            print("", end = "\t")
            if((i+1) % self.m == 0): print("", end="\n")



# cost function class
class COST_FUNCTION:
    def __init__(self, weights = []) -> None:
        self.weights = weights    # 가중치들
        self.avg_r = (weights[-1] - weights[0]) / (len(weights) - 1)    # 평균거리
        self.avg_g = np.array([])    # 그룹 원소들의 평균값
        self.group_range = []    # 그룹들의 범위


    # 가중치를 추가하는 메소드
    def add_weight(self, weight) -> None:
        self.weights = np.append(self.weights, weight)


    # 가중치들을 그룹화하는 메소드
    def avgG(self) -> None:
        groups = [[self.weights[0]]]      # 그룹 초기화


        for i in range(len(self.weights)-1):
            if self.weights[i+1] - self.weights[i] <= self.avg_r:     # 현재 원소와 다음 원소와의 거리가 평균거리보다 짧다면
                groups[-1].append(self.weights[i+1])    # 같은 그룹에 추가
            
            else:
                groups.append([self.weights[i+1]])    # 아니라면 다른 그룹에 추가
        

        single_group_c = len([i for i in groups if len(i) == 1])    # 원소의 개수가 1개인 그룹의 개수

        for group in groups:
            if len(group) > single_group_c:    # 그룹 원소의 개수가 싱글그룹개수보다 많다면
                self.avg_g = np.append(self.avg_g, sum(group) / len(group))    # 그룹 원소들의 평균을 avg_g 에 추가한다.
                self.group_range.append([group[0] - self.avg_r, group[-1] + self.avg_r])    # 그룹 범위를 group_range 에 추가한다.

    # cost method
    def cost(self, weight):
        alpha = self.avg_g
        if len(alpha) == 0: return 0.5

        for i in range(len(alpha) - 1):
            start1 = self.group_range[i][0]  # alpha[i] 구역의 시작점
            end1 = self.group_range[i][1]  # alpha[i] 구역의 끝점

            start2 = self.group_range[i+1][0]  # alpha[i+1] 구역의 시작점


            # weight이 두개의 Cost function 사이에 위치한 경우
            if end1 > start2  and weight >= alpha[i] and weight <= alpha[i+1]:
                a = 5.645 / ((alpha[i+1] - alpha[i])/2 + 0.00001)    # tanh(ax) 에서 x 의 계수
                x = (alpha[i] + alpha[i+1]) / 2    # tanh(x - alpha) 에서 alpha 역할
                b = (np.tanh((alpha[i] - alpha[i+1])/2))**2    # tanh( a(x - alpha) ) + b 에서 b 역할

                return -np.tanh(a * (weight - x))**2 + b - 0.5  # cost


            # weight이 하나의 Cost function 에 위치한 경우
            if weight >= start1 and weight <= end1:
                a = 5.645 / ((end1 - start1) / 2 + 0.00001)    # tanh(ax) 에서 x 의 계수

                return np.tanh(a * (weight - alpha[i]))**2 - 0.5  # cost

        # 마지막 cost function
        start1 = self.group_range[-1][0]
        end1 = self.group_range[-1][1]

        if weight >= start1 and weight <= end1:
            a = 5.645 / ((end1 - start1) / 2 + 0.00001)    # tanh(ax) 에서 x 의 계수

            return np.tanh(a * (weight - alpha[-1]))**2 - 0.5  # cost

        # 5.645 는 tanh^2(x) 의 global minimum(y = 0) 에서 y=1까지의 x 간격이다.


        return 1



# CVIW GROUP class
class CVIW_GROUP:
    def __init__(self, class_name = "", dsize = (128, 128), cviws = []) -> None:
        self.cviws: list = cviws
        self.class_name: str = class_name
        self.important_weight: list = []
        # self.important_weight[i][j] = COST_FUNCTION(weights)
        self.dsize: tuple = dsize


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
            for j in range(8):
                weights = []
                for cviw in self.cviws:
                    weights.append(cviw.weight[i][2])
                
                cost = COST_FUNCTION(sorted(weights))  # cost function 으로 특징가중치를 처리한다.
                cost.avgG()  # 특징가중치 구하는 메소드 호출
                self.important_weight[i].append(cost)  # 특징가중치 리스트에 해당 cost function 클래스 대입
            
            print(f'{i / percent * 100} %', end="\r")

        del self.cviws


    # cost 를 구하는 메소드.
    def cost_function(self, weight):
        cost = 0

        for i in range(self.dsize[0] * self.dsize[1]):
            for j in range(8):
                cost += self.important_weight[i][j].cost(weight[i][j])

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
        cviw_group = CVIW_GROUP(class_name=class_name, dsize = self.dsize)

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
    # Single CVIW Test
    cviw = CVIW(3, 3, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    cviw.Weight_()

    cviw.printWeight()

    cviw2 = CVIW(3, 3, np.array([5, 2, 3, 4, 1, 3, 4, 2, 6]))
    cviw2.Weight_()



    # CVIW Group Test
    cviw_group = CVIW_GROUP(class_name="test", dsize=(3, 3))
    cviw_group.add_cviw([1, 2, 3, 4, 5, 6, 7, 8, 9])
    cviw_group.add_cviw([5, 2, 3, 4, 1, 3, 4, 2, 6])
    # cviw_group.load_add_cviw("file name")

    cviw_group.weight_all()


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
    cost_function.avgG()
    x = linspace(0, 50, 1000)
    y = [cost_function.cost(i) for i in x]
    plt.plot(x, y)
    plt.show()

    cviw_group = CVIW_GROUP(class_name="test", dsize=(3, 3))
    cviw_group.add_cviw([1, 4, 4, 7, 3, 5, 6, 3, 2])
    cviw_group.add_cviw([7, 4, 4, 6, 3, 5, 7, 2, 4])
    cviw_group.add_cviw([14, 6, 2, 4, 6, 7, 6, 7, 8])
    cviw_group.add_cviw([9, 9, 1, 2, 3, 5, 5, 3, 4])
    cviw_group.add_cviw([15, 4, 3, 2, 6, 2, 3, 1, 2])
    cviw_group.add_cviw([25, 8, 7, 2, 4, 2, 23, 4, 2])
    cviw_group.add_cviw([23, 0, 3, 4, 3, 4, 3, 2, 3])
    cviw_group.train_()
