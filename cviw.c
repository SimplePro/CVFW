#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// 0: 오른쪽 위 = m(y-1) + x + 1
// 1: 오른쪽 = my + x + 1
// 2: 오른쪽 아래 = m(y+1) + x + 1
// 3: 아래 = m(y+1) + x
// 4: 왼쪽 아래 = m(y+1) + x - 1
// 5: 왼쪽 = my + x - 1
// 6: 왼쪽 위 = m(y-1) + x - 1
// 7: 위 = m(y-1) + x 

// weight[좌표][위의 방향 중 하나]
double weight[263000][8] = {0};
int P[263000] = {0};


// 첫번째 케이스 가중치 
void case1Weight(int m, int n) {
	int x, y;
	double w_;
	
	for(x = 0; x <= m-2; x++) {
		for(y = 1; y <= n-2; y++) {
			// 오른쪽 위 
			w_ = 1.0 * P[m*(y-1) + x + 1] / P[m*y+x];
			weight[m*y + x][0] = w_;
			weight[m*(y-1) + x + 1][4] = 1 / w_;
			
			// 오른쪽 
			w_ = 1.0 * P[m*y + x + 1] / P[m*y+x];
			weight[m*y + x][1] = w_;
			weight[m*y + x + 1][5] = 1 / w_;
			
			// 오른쪽 아래 
			w_ = 1.0 * P[m*(y+1) + x + 1] / P[m*y+x];
			weight[m*y + x][2] = w_;
			weight[m*(y+1) + x + 1][6] = 1 / w_;
			
			// 아래 
			w_ = 1.0 * P[m*(y+1) + x] / P[m*y+x];
			weight[m*y + x][3] = w_;
			weight[m*(y+1) + x][7] = 1 / w_;
		}
	}
}

// 두번째 케이스 가중치
void case2Weight(int m, int n) {
	int x;
	double w_;
	
	for(x = 0; x <= m-2; x++) {
		// 오른쪽 
		w_ = 1.0 * P[x+1] / P[x];
		weight[x][1] = w_;
		weight[x+1][5] = 1 / w_;
		
		// 오른쪽 아래
		w_ = 1.0 * P[x+m+1] / P[x];
		weight[x][2] = w_;
		weight[x+m+1][6] = 1 / w_;
		
		// 아래
		w_ = 1.0 * P[x+m] / P[x];	
		weight[x][3] = w_;
		weight[x+m][7] = 1 / w_;
	}
}

// 세번째 케이스 가중치
void case3Weight(int m, int n) {
	int x, y = n - 1;
	double w_;
	
	for(x = 0; x <= m-2; x++) {
		// 오른쪽 위
		w_ = 1.0 * P[m*(y-1) + x + 1] / P[m*y + x];
		weight[m*y + x][0] = w_;
		weight[m*(y-1) + x + 1][4] = 1 / w_;
		
		// 오른쪽
		w_ = 1.0 * P[m*y + x + 1] / P[m*y + x];
		weight[m*y + x][1] = w_;
		weight[m*y + x + 1][5] = 1 / w_;
	}
}

// 네번째 케이스 가중치
void case4Weight(int m, int n) {
	int x = m - 1, y;
	double w_;
	
	for(y = 0; y <= n-2; y++) {
		// 아래 
		w_ = 1.0 * P[m*(y+1) + x] / P[m*y + x];
		weight[m*y + x][3] = w_;
		weight[m*(y+1) + x][7] = 1 / w_;
	}
}

void weight_(int m, int n) {
	case1Weight(m, n);
	case2Weight(m, n);
	case3Weight(m, n);
	case4Weight(m, n);
}

void printWeight(int m, int n) {
	int i, j;
	
	for(i = 0; i < m*n; i++) {
		for(j = 0; j < 8; j++) {
			printf("%.2lf ", weight[i][j]);
		}
		printf("  ");
		if((i+1) % m == 0) printf("\n");
	}
}

int main() {
	srand(time(NULL));
	
	int i;
	for(i = 0; i < 512*512; i++) P[i] = rand() % 256;
	
	clock_t start = clock();
	weight_(512, 512);
	printf("Time: %lf\n", (double)(clock() - start) / CLOCKS_PER_SEC);
	
	Sleep(1000);
	
	printWeight(512, 512);
	
	return 0;
}
