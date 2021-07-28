import numpy as np

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


    def case4Weight(self):
        P = self.P
        m = self.m
        x = m - 1

        for y in range(self.n - 1):
            # down
            w_ = P[m * (y + 1) + x] / P[m * y + x]
            self.weight[m * y + x][3] = w_
            self.weight[m * (y + 1) + x][7] = 1 / w_
	

    def printWeight(self):
        for i in range(self.m * self.n):
            for j in range(8):
                print(self.weight[i][j], end=" ")
            
            print("", end = "\t")
            if((i+1) % self.m == 0): print("", end="\n")

cviw = CVIW(3, 3, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
cviw.Weight_()

cviw.printWeight()