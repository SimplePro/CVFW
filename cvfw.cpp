#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <windows.h>

// 0: 오른쪽 위 = m(y-1) + x + 1
// 1: 오른쪽 = my + x + 1
// 2: 오른쪽 아래 = m(y+1) + x + 1
// 3: 아래 = m(y+1) + x
// 4: 왼쪽 아래 = m(y+1) + x - 1
// 5: 왼쪽 = my + x - 1
// 6: 왼쪽 위 = m(y-1) + x - 1
// 7: 위 = m(y-1) + x

// weight[좌표][위의 방향 중 하나]


// 추후에는 힙을 사용하여 SIZE 를 늘린다. 
#define SIZE 25000



// ---------------- start -------------------- CVIW ---------------- start ------------------------
// 하나의 CVIW 클래스 
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


// 첫번째 케이스 가중치 
void CVIW::case1Weight() {
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
void CVIW::case2Weight() {
	double w_;
	
	for(int x = 0; x <= m-2; x++) {
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
void CVIW::case3Weight() {
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
void CVIW::case4Weight() {
	int x = m - 1, y;
	double w_;
	
	for(y = 0; y <= n-2; y++) {
		// 아래 
		w_ = 1.0 * P[m*(y+1) + x] / P[m*y + x];
		weight[m*y + x][3] = w_;
		weight[m*(y+1) + x][7] = 1 / w_;
	}
}


// 가중치 출력 함수 
void CVIW::printWeight() {
	for(int i = 0; i < m*n; i++) {
		for(int j = 0; j < 8; j++) {
			printf("%.2lf ", weight[i][j]);
		}
		printf("  ");
		if((i+1) % m == 0) printf("\n");
	}
}


// 가중치 구하는 함수 
void CVIW::Weight_() {
	if(m == 0 or n == 0) {
		printf("m and n not must be 0. defined it again please.");
		return;
	}
	
	case1Weight();
	case2Weight();
	case3Weight();
	case4Weight();
}


// CVIW 를 Init 하는 메소드 
void CVIW::InitMembers(int m_, int n_, int P_[]) {
	m = m_;
	n = n_;
	for(int i = 0; i < m * n; i++) P[i] = P_[i];
}
// ----------- ----- end ------------------- CVIW ------------------- end -----------------------




// ----------- start --------------------- CVIW_GROUP --------------------- start ---------------------

// 여러 CVIW 를 한번에 관리하는 클래스. 
class CVIW_GROUP {
	public:
		CVIW cviws[1001];
		int idx = 0;
		int size = 0;
		int m, n;
		
		void add_cviw(int P[]);
		void WeightAll();
		void Group();
};

// CVIW 를 추가하는 메소드. 
void CVIW_GROUP::add_cviw(int P[]) {
	if(m == 0 or n == 0) {
		printf("m and n not must be 0. defined it again please.");
		return;
	}
	
	if(m * n != (sizeof(P) / sizeof(int))) {
		printf("size of P must be same from m * n. defined it again please.");
		return;
	}
	
	cviws[idx].InitMembers(m, n, P);
	idx++;
}

// CVIW_GROUP 에 있는 모든 CVIW의 가중치를 한번에 구하는 메소드. 
void CVIW_GROUP::WeightAll() {
	if(size < 1) {
		printf("size not must be smaller than 1. defined it again please.")
		return;
	}
	for(int i = 0; i < size; i++) cviws[i].Weight_();
}

// --------------- end ------------------- CVIW_GROUP ------------------- end -----------------




// ------------ start ---------------------- MAIN --------------------- start --------------------
int main() {
	srand(time(NULL));
	
	CVIW cviw;
	cviw.m = 156; cviw.n = 156;
	
	for(int i = 0; i < 156*156; i++) cviw.P[i] = rand() % 256 + 1;    // P[i] 가 0 인 경우에는 가중치가 무조건 0 이 나오므로 1 을 더해준다. 
	
	
	
	clock_t start = clock();
	cviw.Weight_();
	printf("Time: %lf\n", 1.0 * (clock() - start) / CLOCKS_PER_SEC);
	
	
	Sleep(1000);
	
	cviw.printWeight();
	
	return 0;
}
// ---------------- end ---------------------- MAIN -------------------- end ------------------------
