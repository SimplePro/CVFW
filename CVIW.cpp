#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <windows.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

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



// ---------------- start -------------------- CVIW ---------------- start ------------------------
// �ϳ��� CVIW Ŭ���� 
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


// ù��° ���̽� ����ġ 
void CVIW::case1Weight() {
	int x, y;
	double w_;

	for (x = 0; x <= m - 2; x++) {
		for (y = 1; y <= n - 2; y++) {
			// ������ �� 
			w_ = 1.0 * P[m * (y - 1) + x + 1] / P[m * y + x];
			weight[m * y + x][0] = w_;
			weight[m * (y - 1) + x + 1][4] = 1 / w_;

			// ������ 
			w_ = 1.0 * P[m * y + x + 1] / P[m * y + x];
			weight[m * y + x][1] = w_;
			weight[m * y + x + 1][5] = 1 / w_;

			// ������ �Ʒ� 
			w_ = 1.0 * P[m * (y + 1) + x + 1] / P[m * y + x];
			weight[m * y + x][2] = w_;
			weight[m * (y + 1) + x + 1][6] = 1 / w_;

			// �Ʒ� 
			w_ = 1.0 * P[m * (y + 1) + x] / P[m * y + x];
			weight[m * y + x][3] = w_;
			weight[m * (y + 1) + x][7] = 1 / w_;
		}
	}
}


// �ι�° ���̽� ����ġ		
void CVIW::case2Weight() {
	double w_;

	for (int x = 0; x <= m - 2; x++) {
		// ������ 
		w_ = 1.0 * P[x + 1] / P[x];
		weight[x][1] = w_;
		weight[x + 1][5] = 1 / w_;

		// ������ �Ʒ�
		w_ = 1.0 * P[x + m + 1] / P[x];
		weight[x][2] = w_;
		weight[x + m + 1][6] = 1 / w_;

		// �Ʒ�
		w_ = 1.0 * P[x + m] / P[x];
		weight[x][3] = w_;
		weight[x + m][7] = 1 / w_;
	}
}


// ����° ���̽� ����ġ		
void CVIW::case3Weight() {
	int x, y = n - 1;
	double w_;

	for (x = 0; x <= m - 2; x++) {
		// ������ ��
		w_ = 1.0 * P[m * (y - 1) + x + 1] / P[m * y + x];
		weight[m * y + x][0] = w_;
		weight[m * (y - 1) + x + 1][4] = 1 / w_;

		// ������
		w_ = 1.0 * P[m * y + x + 1] / P[m * y + x];
		weight[m * y + x][1] = w_;
		weight[m * y + x + 1][5] = 1 / w_;
	}
}


// �׹�° ���̽� ����ġ
void CVIW::case4Weight() {
	int x = m - 1, y;
	double w_;

	for (y = 0; y <= n - 2; y++) {
		// �Ʒ� 
		w_ = 1.0 * P[m * (y + 1) + x] / P[m * y + x];
		weight[m * y + x][3] = w_;
		weight[m * (y + 1) + x][7] = 1 / w_;
	}
}


// ����ġ ��� �Լ� 
void CVIW::printWeight() {
	for (int i = 0; i < m * n; i++) {
		for (int j = 0; j < 8; j++) {
			printf("%.2lf ", weight[i][j]);
		}
		printf("  ");
		if ((i + 1) % m == 0) printf("\n");
	}
}


// ����ġ ���ϴ� �Լ� 
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


// CVIW �� Init �ϴ� �޼ҵ� 
void CVIW::InitMembers(int m_, int n_, int P_[]) {
	m = m_;
	n = n_;
	for (int i = 0; i < m * n; i++) P[i] = P_[i];
}
// ----------- ----- end ------------------- CVIW ------------------- end -----------------------




// ----------- start --------------------- CVIW_GROUP --------------------- start ---------------------

// ���� CVIW �� �ѹ��� �����ϴ� Ŭ����. 
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

// CVIW �� �߰��ϴ� �޼ҵ�. 
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

// CVIW_GROUP �� �ִ� ��� CVIW�� ����ġ�� �ѹ��� ���ϴ� �޼ҵ�. 
void CVIW_GROUP::WeightAll() {
	if (size < 1) {
		printf("size not must be smaller than 1. defined it again please.");
		return;
	}
	for (int i = 0; i < size; i++) cviws[i].Weight_();
}

// --------------- end ------------------- CVIW_GROUP ------------------- end -----------------




// ------------ start ---------------------- MAIN --------------------- start --------------------
int main() {
	srand(time(NULL));

	CVIW cviw;
	cviw.m = 156; cviw.n = 156;

	for (int i = 0; i < 156 * 156; i++) cviw.P[i] = rand() % 256 + 1;    // P[i] �� 0 �� ��쿡�� ����ġ�� ������ 0 �� �����Ƿ� 1 �� �����ش�. 



	clock_t start = clock();
	cviw.Weight_();
	printf("Time: %lf\n", 1.0 * (clock() - start) / CLOCKS_PER_SEC);


	Sleep(1000);

	cviw.printWeight();

	
	// cv library test
	Mat female1 = imread("C:\\Users\\SimplePro\\Downloads\\female1.jpg", 0);
	imshow("female1", female1);

	Mat female2 = imread("C:\\Users\\SimplePro\\Downloads\\female2.jpg", 0);
	imshow("female2", female2);

	Mat male1 = imread("C:\\Users\\SimplePro\\Downloads\\male1.jpg", 0);
	imshow("male1", male1);

	Mat male2 = imread("C:\\Users\\SimplePro\\Downloads\\male2.jpg", 0);
	imshow("male2", male2);
	waitKey(0);

	return 0;
}
// ---------------- end ---------------------- MAIN -------------------- end ------------------------