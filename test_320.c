/* Test program for analyzing performance around size 320 */
#include <stdio.h>
#include <stdlib.h>
#include "parameters.h"

void MY_MMult(int, int, int, double *, int, double *, int, double *, int );
double dclock();

int main() {
    int sizes[] = {318, 319, 320, 321, 322};
    int n_sizes = 5;
    
    for (int i = 0; i < n_sizes; i++) {
        int p = sizes[i];
        int m = p, n = p, k = p;
        int lda = m, ldb = k, ldc = m;
        
        double *a = malloc(lda * k * sizeof(double));
        double *b = malloc(ldb * n * sizeof(double));
        double *c = malloc(ldc * n * sizeof(double));
        
        // Initialize matrices
        for (int j = 0; j < lda * k; j++) a[j] = rand() / (double)RAND_MAX;
        for (int j = 0; j < ldb * n; j++) b[j] = rand() / (double)RAND_MAX;
        for (int j = 0; j < ldc * n; j++) c[j] = rand() / (double)RAND_MAX;
        
        // Time the multiplication
        double dtime = dclock();
        MY_MMult(m, n, k, a, lda, b, ldb, c, ldc);
        dtime = dclock() - dtime;
        
        double gflops = 2.0 * m * n * k * 1.0e-09 / dtime;
        printf("Size: %d, Performance: %le GFLOPS\n", p, gflops);
        
        free(a); free(b); free(c);
    }
    return 0;
} 