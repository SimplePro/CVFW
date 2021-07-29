import numpy as np
import cv2


class CVIW:
    def __init__(self, m, n, P) -> None:
        self.m = m
        self.n = n
        self.weight = np.zeros([50000, 8])
        self.P = P
    
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


# cost method
def cost(weight1, weight2):
    return sum(sum(abs(weight1 - weight2)))  # 차이의 합



# CVIW GROUP class
class CVIW_GROUP:
    def __init__(self, class_name = "", cviws = [], dsize = (128, 128)) -> None:
        self.cviws: list = cviws
        self.class_name: str = class_name
        self.important_weight: list = []
        self.dsize: tuple = dsize

    # add cviw method, img is flatten list of pixel data to add.
    def add_cviw(self, img) -> None:
        cviw = CVIW(self.dsize[0], self.dsize[1], img)
        cviw.Weight_()
        self.cviws.append(cviw)


    # all cviw in cviws list get weight.
    def weight_all(self) -> None:
        for i in range(len(self.cviws)):
            self.cviws[i].Weight_()

    
    # load then add cviw method.
    def load_add_cviw(self, file):
        img = cv2.resize(cv2.cvtColor(cv2.imread(file, 1), cv2.COLOR_BGR2GRAY), dsize=self.dsize).flatten().tolist()
        cviw = CVIW(self.dsize[0], self.dsize[1], img)
        cviw.Weight_()
        self.add_cviw(cviw)


# load image method
def load(file, dsize_ = (128, 128)) -> list:
    img = cv2.resize(cv2.cvtColor(cv2.imread(file, 1), cv2.COLOR_BGR2GRAY), dsize=dsize_)
    img = img.flatten().tolist()

    return img



if __name__ == '__main__':
    # single CVIW Test
    cviw = CVIW(3, 3, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    cviw.Weight_()

    cviw.printWeight()

    cviw2 = CVIW(3, 3, np.array([5, 2, 3, 4, 1, 3, 4, 2, 6]))
    cviw2.Weight_()

    print(f"cost: {cost(cviw.weight, cviw2.weight)}")



    # CVIW Group Test
    cviw_group = CVIW_GROUP(class_name="test", dsize=(3, 3))
    cviw_group.add_cviw([1, 2, 3, 4, 5, 6, 7, 8, 9])
    cviw_group.add_cviw([5, 2, 3, 4, 1, 3, 4, 2, 6])

    cviw_group.weight_all()

    print(f"cost: {cost(cviw_group.cviws[0].weight, cviw_group.cviws[1].weight)}")