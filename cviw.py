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


# cost function class
class COST_FUNCTION:
    def __init__(self, weights = []) -> None:
        self.weights = weights
        self.avg_r = (weights[-1] - weights[0]) / (len(weights) - 1)
        self.avg_g = np.array([])

    def avgG(self) -> None:
        groups = [[self.weights[0]]]

        for i in range(len(self.weights)-1):
            if self.weights[i+1] - self.weights[i] <= self.avg_r:
                groups[-1].append(self.weights[i+1])
            
            else: groups.append([self.weights[i+1]])

        single_group_c = len([i for i in groups if len(i) == 1])
        for group in groups:
            if len(group) > single_group_c:
                self.avg_g = np.append(self.avg_g, sum(group) / len(group))

    
    def cost(self, weight):
        for alpha in self.avg_g:
            if weight >= alpha - 3 and weight <= alpha + 3: return (np.tanh(weight - alpha))**2

        return 1
            
        
a = COST_FUNCTION(weights=[1, 2, 4, 6, 9])
a.avgG()
print(a.avg_g)
# print(a.cost())

# CVIW GROUP class
class CVIW_GROUP:
    def __init__(self, class_name = "", dsize = (128, 128), cviws = []) -> None:
        self.cviws: list = cviws
        self.class_name: str = class_name
        self.important_weight: list = np.zeros([50000, 8])
        # self.important_weight[i][j] = COST_FUNCTION(weights)
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



if __name__ == '__main__':
    # single CVIW Test
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
