import numpy as np
import cv2
from numpy.core.function_base import linspace
from os import listdir
from tqdm import tqdm


# Weight method
def Weight(m, n, P):
    weight = np.zeros([m*n, 2])
    P = ((np.array(P) + 1) / 256.0).tolist()


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
        self.fw_count = 0
        self.fw = []  # 특징 그룹의 원소들의 평균값
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

        if count != 1:
            self.mean_groups.append(S / count)
            self.group_c.append(count)
    
        self.fw = []

        self.fw_count = 1

        if len(self.mean_groups) == 0: self.fw_count = 0
        else:
            argsort_ = np.argsort(self.group_c)[::-1][:self.feature_group_number]
            for i in argsort_:
                if self.group_c[i] > self.feature_weight_number:
                    self.fw.append(self.mean_groups[i])

        self.weights = []  # 메모리 비움.


    # cost method
    def cost_function(self, weight):
        if len(self.fw) == 0: return 0
        return np.min((self.fw - weight)**2)

    
    # update method
    def update__(self, feature_group_number, feature_weight_number):
        self.fw = []

        self.fw_count = 1

        if len(self.mean_groups) == 0: self.fw_count = 0

        else:
            argsort_ = np.argsort(self.group_c)[::-1][:feature_group_number]
            for i in argsort_:
                if self.group_c[i] > feature_weight_number:
                    self.fw.append(self.mean_groups[i])



# CVFW GROUP class
class CVFW_GROUP:
    def __init__(self, class_name = "", dsize = (128, 128)) -> None:
        self.cvfws_weight: list = []
        self.class_name: str = class_name
        self.feature_weight: list = []
        # self.feature_weight[i][j] = COST_FUNCTION(weights, feature_group_number, feature_weight_number)
        self.dsize: tuple = dsize
        self.fw_count = 0  # 특징 가중치의 개수

    
    def add_cvfw_weight(self, img) -> None:
        self.cvfws_weight.append(Weight(self.dsize[0], self.dsize[1], img))

    
    # load then add cvfw method.
    def load_add_cvfw(self, file):
        img = cv2.resize(cv2.cvtColor(cv2.imread(file, 1), cv2.COLOR_BGR2GRAY), dsize=self.dsize).flatten().tolist()  # 로드한 이미지를 flatten 한다.
        self.cvfws_weight.append(Weight(self.dsize[0], self.dsize[1], img))

    
    # train method.
    def train_(self, feature_group_number = 3, feature_weight_number = 50):
        
        for i in tqdm(range(self.dsize[0] * self.dsize[1]), desc=self.class_name, mininterval=1):
            self.feature_weight.append([])
            for j in range(2):
                weights = []
                for w in self.cvfws_weight:
                    weights.append(w[i][j])
                
                cost = COST_FUNCTION(sorted(weights), feature_group_number, feature_weight_number)  # cost function 으로 특징가중치를 처리한다.
                cost.feature_weight()  # 특징 그룹을 구하는 메소드.
                self.fw_count += cost.fw_count  # 특징가중치의 존재 여부(0, 1) 를 fw count 에 더한다.
                self.feature_weight[i].append(cost)  # 특징가중치 리스트에 해당 cost function 클래스 대입
        
        self.cvfws_weight = []  # 메모리 비움.

    
    # update method
    def update_(self, feature_group_number, feature_weight_number):
        self.fw_count = 0

        for i in range(self.dsize[0] * self.dsize[1]):
            for j in range(2):
                self.feature_weight[i][j].update__(feature_group_number, feature_weight_number)
                self.fw_count += self.feature_weight[i][j].fw_count


    # update cost method
    # def update_cost(self, weight):
    #     for i in range(self.dsize[0] * self.dsize[1]):
    #         for j in range(2):
    #             pass
                

    # cost 를 구하는 메소드.
    def cost(self, weight):
        return sum([self.feature_weight[i][j].cost_function(weight[i][j]) for i in range(self.dsize[0] * self.dsize[1]) for j in range(2)]) / self.fw_count

    
    # 가장 대표적인 특징을 담은 img를 모델링해서 반환하는 메소드.
    def modeling_(self, start = 1):
        img = [0 for _ in range(self.dsize[0] * self.dsize[1])]
        img[0] = start
        sum_c = 0
        feature_c = 0

        for i in range(1, self.dsize[0]):  # 맨 위 가로줄
            right_mean_groups = self.feature_weight[i-1][0].mean_groups
            right_group_c = self.feature_weight[i-1][0].group_c
            right_weight = right_mean_groups[right_group_c.index(max(right_group_c))]

            sum_c += sum(right_group_c) - max(right_group_c)
            feature_c += max(right_group_c)

            img[i] = right_weight * img[i-1]


        for i in range(self.dsize[0], self.dsize[1] * self.dsize[0], self.dsize[0]):  # 왼쪽 세로줄
            down_mean_groups = self.feature_weight[i - self.dsize[0]][1].mean_groups
            down_group_c = self.feature_weight[i - self.dsize[0]][1].group_c
            down_weight = down_mean_groups[down_group_c.index(max(down_group_c))]

            sum_c += sum(down_group_c) - max(down_group_c)
            feature_c += max(down_group_c)

            img[i] = down_weight * img[i - self.dsize[0]]
            

        for x in range(1, self.dsize[0]):
            for y in range(1, self.dsize[1]):
                xy = self.dsize[0] * y + x

                m = self.dsize[0]

                right_mean_groups = self.feature_weight[xy-1][0].mean_groups
                right_group_c = self.feature_weight[xy-1][0].group_c
                right_weight = right_mean_groups[right_group_c.index(max(right_group_c))]
                max_right_group_c = max(right_group_c)
                
                down_mean_groups = self.feature_weight[xy - m][1].mean_groups
                down_group_c = self.feature_weight[xy - m][1].group_c
                down_weight = down_mean_groups[down_group_c.index(max(down_group_c))]
                max_down_group_c = max(down_group_c)

                sum_c += sum(right_group_c) + sum(down_group_c) - max_right_group_c - max_down_group_c
                feature_c += max_right_group_c + max_down_group_c

                img[xy] = (img[xy-1] * right_weight * max_right_group_c + img[xy - m] * down_weight * max_down_group_c) / (max_right_group_c + max_down_group_c)

        img = np.array(img).reshape(self.dsize[1], self.dsize[0])

        print("(feature_count) / (sum count) :", feature_c / sum_c)

        return img


# CVFW 모델 클래스
class CVFW_MODEL:
    def __init__(self, dsize = (128, 128), feature_group_number = 3, feature_weight_number = 50) -> None:
        self.cvfw_groups = []
        self.classes = []
        self.dsize = dsize
        self.feature_group_number = feature_group_number
        self.feature_weight_number = feature_weight_number

    
    # 디렉토리 경로로 CVFW_GROUP 을 추가하는 메소드.
    def add_directory(self, class_name = "", path = "") -> None:
        self.classes.append(class_name)
        imgs = listdir(path)
        cvfw_group = CVFW_GROUP(class_name=class_name, dsize = self.dsize)

        for img in imgs:
            cvfw_group.load_add_cvfw(file = f'{path}\\{img}')

        self.cvfw_groups.append(cvfw_group)


    # 학습하는 메소드
    def train(self) -> None:
        for i in range(len(self.classes)):
            self.cvfw_groups[i].train_(self.feature_group_number, self.feature_weight_number)


    # 클래스를 예측하는 메소드
    def predict_class(self, img):
        cvfw_weight = Weight(self.dsize[0], self.dsize[1], img)

        cost_list = []

        for cvfw_group in self.cvfw_groups:
            cost_list.append(cvfw_group.cost(cvfw_weight))

        sum_cost_list = sum(cost_list)
        predict = []
        for i in cost_list:
            predict.append(i / sum_cost_list)

        return predict


    # 각 클래스의 가장 큰 특징을 나타내는 이미지의 유사도를 측정하여 클래스를 예측하는 메소드
    # 하지만 cost 들의 값이 전부 비슷하기 때문에 추천되는 예측 방법이 아니다.
    def modeling_predict_class(self, img):
        cvfw_weight = Weight(self.dsize[0], self.dsize[1], img)
        
        class_weights = []

        for class_name in self.classes:
            modeling_img = self.modeling(class_name).flatten().tolist()
            modeling_weight = Weight(self.dsize[0], self.dsize[1], modeling_img)
            class_weights.append(modeling_weight)


        class_cost = []

        for class_weight in class_weights:
            class_cost.append(sum(sum(abs(class_weight - cvfw_weight))))
        
        sum_cost_list = sum(class_cost)
        predict = []
        for cost in class_cost:
            predict.append(cost / sum_cost_list)

        return predict



    def modeling(self, class_name = "", start = 1):
        cvfw_group = self.cvfw_groups[self.classes.index(class_name)]
        img = cvfw_group.modeling_(start)
        return img



# CVFW Update Class
class CVFW_UPDATE:
    def __init__(self, cvfw_model, feature_group_number = [3], feature_weight_number = [50]) -> None:
        self.cvfw_model = cvfw_model
        self.feature_group_number = feature_group_number  # 하나의 가중치당 feature_group 의 최대 개수
        self.feature_weight_number = feature_weight_number  # 
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
                for i in range(len(self.cvfw_model.cvfw_groups)):
                    self.cvfw_model.cvfw_groups[i].update_(feature_group_number = fgn, feature_weight_number = fwn)
                
                self.accuracy_list.append(self.accuracy())
                print(f"feature_group_number: {fgn}, feature_weight_nummber: {fwn} Done!, accuracy: {self.accuracy_list[-1]}")

                
    # accuracy method
    def accuracy(self):
        true = 0
        count = 0

        for i in range(len(self.validation_path)):
            for file in listdir(self.validation_path[i]):
                img = cv2.resize(cv2.cvtColor(cv2.imread(f"{self.validation_path[i]}\\{file}", 1), cv2.COLOR_BGR2GRAY), dsize=self.cvfw_model.dsize).flatten().tolist()
                predict = self.cvfw_model.predict_class(img)
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

    cvfw_group = CVFW_GROUP(class_name="test", dsize=(128, 128))
    for i in range(500):
        cvfw_group.add_cvfw_weight([randint(1, 256) for _ in range(128 * 128)])

    start_time = time()
    cvfw_group.train_()
    print(f"time: {time() - start_time}")


    # Cost Function
    cost_function = COST_FUNCTION(weights = [1, 7, 9, 14, 15, 23, 27])
    cost_function.feature_weight()
    x = linspace(0, 50, 1000)
    y = [cost_function.cost_function(i) for i in x]
    plt.plot(x, y)
    plt.show()
