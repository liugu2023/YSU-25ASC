/* Create macros so that the matrices are stored in column-major order */
#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

#define min(i, j) ((i) < (j) ? (i) : (j))
#define BLOCK_SIZE 32
#define SMALL_BLOCK_SIZE 8   // 内层小块大小

/* 计算小块矩阵乘法 */
static void do_block(int M, int N, int K,
                    double *a, int lda, int start_a_i, int start_a_j,
                    double *b, int ldb, int start_b_i, int start_b_j,
                    double *c, int ldc, int start_c_i, int start_c_j)
{
    for (int i = 0; i < M; i += SMALL_BLOCK_SIZE) {
        int i_end = min(M, i + SMALL_BLOCK_SIZE);
        for (int j = 0; j < N; j += SMALL_BLOCK_SIZE) {
            int j_end = min(N, j + SMALL_BLOCK_SIZE);
            
            // 预取C块的数据
            double c_block[SMALL_BLOCK_SIZE][SMALL_BLOCK_SIZE] = {0};
            
            // 对K维度分块计算
            for (int k = 0; k < K; k++) {
                for (int ii = i; ii < i_end; ii++) {
                    double a_val = A(start_a_i + ii, start_a_j + k);
                    for (int jj = j; jj < j_end; jj++) {
                        c_block[ii-i][jj-j] += a_val * B(start_b_i + k, start_b_j + jj);
                    }
                }
            }
            
            // 写回结果
            for (int ii = i; ii < i_end; ii++) {
                for (int jj = j; jj < j_end; jj++) {
                    C(start_c_i + ii, start_c_j + jj) += c_block[ii-i][jj-j];
                }
            }
        }
    }
}

/* Routine for computing C = A * B + C */
void MY_MMult(int m, int n, int k, double *a, int lda,
              double *b, int ldb, double *c, int ldc)
{
    // 外层大块分块
    for (int i = 0; i < m; i += BLOCK_SIZE) {
        int M = min(BLOCK_SIZE, m - i);
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            int N = min(BLOCK_SIZE, n - j);
            for (int p = 0; p < k; p += BLOCK_SIZE) {
                int K = min(BLOCK_SIZE, k - p);
                
                // 处理当前块
                do_block(M, N, K,
                        a, lda, i, p,
                        b, ldb, p, j,
                        c, ldc, i, j);
            }
        }
    }
} 