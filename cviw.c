#include <stdio.h>
#include <stdlib.h>



// 0: 오른쪽 위 = m(y-1) + x + 1
// 1: 오른쪽 = m(y-1) + x + 1
// 2: 오른쪽 아래 = m(y+1) + x + 1
// 3: 아래 = m(y+1) + x

// weight[좌표][위의 방향 중 하나]
int weight[16400][4] = {0};
int P[128][128] = {0};


void weight_(int m, int n) {
	
}

int main() {
	printf("Hello, World!\n");
	return 0;
}
