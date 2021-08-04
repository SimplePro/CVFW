#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <windows.h>
#include <math.h>
#include <fstream>
#include <string>

using namespace std;

// 0: right up = m(y-1) + x + 1
// 1: right = my + x + 1
// 2: right down = m(y+1) + x + 1
// 3: down = m(y+1) + x
// 4: left down = m(y+1) + x - 1
// 5: left = my + x - 1
// 6: left up = m(y-1) + x - 1
// 7: up = m(y-1) + x

// weight[index][0 ~ 7]


#define SIZE 17000

// ---------------- start -------------------- CVIW ---------------- start ------------------------
// CVIW class
class CVIW {
public:
	int m;
	int n;
	double weight[SIZE][8] = { {0} };
	int P[SIZE];

	void InitMembers(int m, int n, int P_[]);
	void Weight_();
	void case1Weight();
	void case2Weight();
	void case3Weight();
	void case4Weight();
	void printWeight();

};


// first case Weight
void CVIW::case1Weight() {
	int x, y;
	double w_;

	for (x = 0; x <= m - 2; x++) {
		for (y = 1; y <= n - 2; y++) {
			// right up
			w_ = 1.0 * P[m * (y - 1) + x + 1] / P[m * y + x];
			weight[m * y + x][0] = w_;
			weight[m * (y - 1) + x + 1][4] = 1 / w_;

			// right
			w_ = 1.0 * P[m * y + x + 1] / P[m * y + x];
			weight[m * y + x][1] = w_;
			weight[m * y + x + 1][5] = 1 / w_;

			// right down
			w_ = 1.0 * P[m * (y + 1) + x + 1] / P[m * y + x];
			weight[m * y + x][2] = w_;
			weight[m * (y + 1) + x + 1][6] = 1 / w_;

			// down
			w_ = 1.0 * P[m * (y + 1) + x] / P[m * y + x];
			weight[m * y + x][3] = w_;
			weight[m * (y + 1) + x][7] = 1 / w_;
		}
	}
}


// second case Weight	
void CVIW::case2Weight() {
	double w_;

	for (int x = 0; x <= m - 2; x++) {
		// right
		w_ = 1.0 * P[x + 1] / P[x];
		weight[x][1] = w_;
		weight[x + 1][5] = 1 / w_;

		// right down
		w_ = 1.0 * P[x + m + 1] / P[x];
		weight[x][2] = w_;
		weight[x + m + 1][6] = 1 / w_;

		// down
		w_ = 1.0 * P[x + m] / P[x];
		weight[x][3] = w_;
		weight[x + m][7] = 1 / w_;
	}
}


// third case Weight
void CVIW::case3Weight() {
	int x, y = n - 1;
	double w_;

	for (x = 0; x <= m - 2; x++) {
		// right up
		w_ = 1.0 * P[m * (y - 1) + x + 1] / P[m * y + x];
		weight[m * y + x][0] = w_;
		weight[m * (y - 1) + x + 1][4] = 1 / w_;

		// right
		w_ = 1.0 * P[m * y + x + 1] / P[m * y + x];
		weight[m * y + x][1] = w_;
		weight[m * y + x + 1][5] = 1 / w_;
	}
}


// fourth case Weight
void CVIW::case4Weight() {
	int x = m - 1, y;
	double w_;

	for (y = 0; y <= n - 2; y++) {
		// down
		w_ = 1.0 * P[m * (y + 1) + x] / P[m * y + x];
		weight[m * y + x][3] = w_;
		weight[m * (y + 1) + x][7] = 1 / w_;
	}
}


// print Weight 
void CVIW::printWeight() {
	for (int i = 0; i < m * n; i++) {
		for (int j = 0; j < 8; j++) {
			printf("%.2lf ", weight[i][j]);
		}
		printf("  ");
		if ((i + 1) % m == 0) printf("\n");
	}
}


// Weight
void CVIW::Weight_() {
	if (m == 0 or n == 0) {
		printf("m and n not must be 0. defined it again please.");
		return;
	}

	case1Weight();
	case2Weight();
	case3Weight();
	case4Weight();
}


// CVIW Init Members
void CVIW::InitMembers(int m_, int n_, int P_[]) {
	m = m_;
	n = n_;
	for (int i = 0; i < m * n; i++) P[i] = P_[i] + 1;
}
// ----------- ----- end ------------------- CVIW ------------------- end -----------------------




// ----------- start --------------------- CVIW_GROUP --------------------- start ---------------------

// CVIW GROUP CLASS
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

// add CVIW
void CVIW_GROUP::add_cviw(int P[]) {
	if (m == 0 or n == 0) {
		printf("m and n not must be 0. defined it again please.");
		return;
	}

	if (m * n != (sizeof(P) / sizeof(int))) {
		printf("size of P must be same from m * n. defined it again please.");
		return;
	}

	cviws[idx].InitMembers(m, n, P);
	idx++;
}

// all the CVIW in CVIW_GROUP are weight extract.
void CVIW_GROUP::WeightAll() {
	if (size < 1) {
		printf("size not must be smaller than 1. defined it again please.");
		return;
	}
	for (int i = 0; i < size; i++) cviws[i].Weight_();
}

// --------------- end ------------------- CVIW_GROUP ------------------- end -----------------



double cost(CVIW cviw1, CVIW cviw2, int m, int n) {
	double c = 0.0;

	for (int i = 0; i < m * n; i++) {
		for (int j = 0; j < 8; j++) {
			c += abs(cviw1.weight[i][j] - cviw2.weight[i][j]);
		}
	}

	return c;
}


// ------------ start ---------------------- MAIN --------------------- start --------------------
int main() {
	srand(time(NULL));


	CVIW cviw1, cviw2;

	cviw1.m = 128; cviw1.n = 128;
	cviw2.m = 128; cviw2.n = 128;

	ifstream fin;

	char line[200];

	fin.open("C:\\kimdonghwan\\python\\CVIW\\male2.txt");
	if (fin.is_open()) {
		int i = 0;
		while (fin.getline(line, sizeof(line))) {
			cviw1.P[i] = atoi(line) + 1;
			i++;
		}
		cviw1.Weight_();
		printf("\n");
	}
	fin.close();

	//for (int i = 0; i < 128 * 128; i++) cviw1.P[i] = rand() % 256 + 1;
	//cviw1.Weight_();


	fin.open("C:\\kimdonghwan\\python\\CVIW\\male1.txt");
	if (fin.is_open()) {
		int i = 0;
		while (fin.getline(line, sizeof(line))) {
			cviw2.P[i] = atoi(line) + 1;
			i++;
		}
		cviw2.Weight_();
		printf("\n");

	}
	fin.close();

	//for (int i = 0; i < 128 * 128; i++) cviw2.P[i] = rand() % 256 + 1;
	//cviw2.Weight_();

	double c = cost(cviw1, cviw2, 128, 128);
	printf("%lf", c);



	//	clock_t start = clock();
	//	cviw.Weight_();
	//	printf("Time: %lf\n", 1.0 * (clock() - start) / CLOCKS_PER_SEC);


	//	Sleep(1000);

	//	cviw.printWeight();

	return 0;
}
// ---------------- end ---------------------- MAIN -------------------- end ------------------------
