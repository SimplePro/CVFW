#include <iostream>


using namespace std;

float weight[1000][1000] = {};    // i_w_j = w[i][j]
int P[1000][1000] = {};    // P(n, m) = P[n][m]


// m 은 가로, n 은 세로 
void weight_(int m, int n) {
	int i, j;
	// x = 0 ~ m,    y = 0, n     (가로변에 해당하는 픽셀들) 
	for(i = 0; i <= n; i += n - 1) {
		for(j = 0; j < m; j++) {
			continue; 
		}
	}
	
	
	// y = 1 ~ n - 1,    x = 0, m     (세로변에 해당하는 픽셀들) 
	for(i = 0; i <= m; i += m - 1) {
		for(j = 0; j < n; j++) {
			continue;
		}
	}
	
	
	// 나머지 안쪽 부분. 
	for(i = 1; i < m - 1; i++) {
		for(j = 1; j < n - 1; j++) {
			continue;
		}
	}
}

int main() {
	cout << "Hello, World!";
	P[0][0] = 10;
	weight_(15, 10);
	return 0;
}
