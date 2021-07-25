#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <windows.h>

// 0: ������ �� = m(y-1) + x + 1
// 1: ������ = my + x + 1
// 2: ������ �Ʒ� = m(y+1) + x + 1
// 3: �Ʒ� = m(y+1) + x
// 4: ���� �Ʒ� = m(y+1) + x - 1
// 5: ���� = my + x - 1
// 6: ���� �� = m(y-1) + x - 1
// 7: �� = m(y-1) + x

// weight[��ǥ][���� ���� �� �ϳ�]


// ���Ŀ��� ���� ����Ͽ� SIZE �� �ø���. 
#define SIZE 25000

class CVIW {
	public:
		int m;
		int n;
		double weight[SIZE][8] = {{0}};
		int P[SIZE];
		
		void InitMembers(int m, int n, int P_[]);
		void Weight_();
		void case1Weight();
		void case2Weight();
		void case3Weight();
		void case4Weight();
		void printWeight();
		
};


// ù��° ���̽� ����ġ 
void CVIW::case1Weight() {
	int x, y;
	double w_;
	
	for(x = 0; x <= m-2; x++) {
		for(y = 1; y <= n-2; y++) {
			// ������ �� 
			w_ = 1.0 * P[m*(y-1) + x + 1] / P[m*y+x];
			weight[m*y + x][0] = w_;
			weight[m*(y-1) + x + 1][4] = 1 / w_;
			
			// ������ 
			w_ = 1.0 * P[m*y + x + 1] / P[m*y+x];
			weight[m*y + x][1] = w_;
			weight[m*y + x + 1][5] = 1 / w_;
			
			// ������ �Ʒ� 
			w_ = 1.0 * P[m*(y+1) + x + 1] / P[m*y+x];
			weight[m*y + x][2] = w_;
			weight[m*(y+1) + x + 1][6] = 1 / w_;
			
			// �Ʒ� 
			w_ = 1.0 * P[m*(y+1) + x] / P[m*y+x];
			weight[m*y + x][3] = w_;
			weight[m*(y+1) + x][7] = 1 / w_;
		}
	}
}


// �ι�° ���̽� ����ġ		
void CVIW::case2Weight() {
	double w_;
	
	for(int x = 0; x <= m-2; x++) {
		// ������ 
		w_ = 1.0 * P[x+1] / P[x];
		weight[x][1] = w_;
		weight[x+1][5] = 1 / w_;
		
		// ������ �Ʒ�
		w_ = 1.0 * P[x+m+1] / P[x];
		weight[x][2] = w_;
		weight[x+m+1][6] = 1 / w_;
		
		// �Ʒ�
		w_ = 1.0 * P[x+m] / P[x];	
		weight[x][3] = w_;
		weight[x+m][7] = 1 / w_;
	}
}


// ����° ���̽� ����ġ		
void CVIW::case3Weight() {
	int x, y = n - 1;
	double w_;
	
	for(x = 0; x <= m-2; x++) {
		// ������ ��
		w_ = 1.0 * P[m*(y-1) + x + 1] / P[m*y + x];
		weight[m*y + x][0] = w_;
		weight[m*(y-1) + x + 1][4] = 1 / w_;
		
		// ������
		w_ = 1.0 * P[m*y + x + 1] / P[m*y + x];
		weight[m*y + x][1] = w_;
		weight[m*y + x + 1][5] = 1 / w_;
	}
}


// �׹�° ���̽� ����ġ
void CVIW::case4Weight() {
	int x = m - 1, y;
	double w_;
	
	for(y = 0; y <= n-2; y++) {
		// �Ʒ� 
		w_ = 1.0 * P[m*(y+1) + x] / P[m*y + x];
		weight[m*y + x][3] = w_;
		weight[m*(y+1) + x][7] = 1 / w_;
	}
}


// ����ġ ��� �Լ� 
void CVIW::printWeight() {
	for(int i = 0; i < m*n; i++) {
		for(int j = 0; j < 8; j++) {
			printf("%.2lf ", weight[i][j]);
		}
		printf("  ");
		if((i+1) % m == 0) printf("\n");
	}
}


// ����ġ ���ϴ� �Լ� 
void CVIW::Weight_() {
	case1Weight();
	case2Weight();
	case3Weight();
	case4Weight();
}


void CVIW::InitMembers(int m_, int n_, int P_[]) {
	m = m_;
	n = n_;
	for(int i = 0; i < m * n; i++) P[i] = P_[i];
}


int main() {
	srand(time(NULL));
	
	CVIW cviw1;
	cviw1.m = 128; cviw1.n = 128;
	
	for(int i = 0; i < 128*128; i++) cviw1.P[i] = rand() % 256 + 1;    // P[i] �� 0 �� ��쿡�� ����ġ�� ������ 0 �� �����Ƿ� 1 �� �����ش�. 
	
	
	
	clock_t start = clock();
	cviw1.Weight_();
	printf("Time: %lf\n", 1.0 * (clock() - start) / CLOCKS_PER_SEC);
	
	
	Sleep(1000);
	
	cviw1.printWeight();
	
	return 0;
}
