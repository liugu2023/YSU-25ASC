/* Create macros so that the matrices are stored in column-major order */
#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

#define min(i, j) ((i) < (j) ? (i) : (j))
#define BLOCK_SIZE 48        // 240 = 48 * 5
#define SMALL_BLOCK_SIZE 12  // 48 = 12 * 4
#define VECTOR_SIZE 4        // AVX2 = 4 doubles
#define ALIGN_TO 32

#include <immintrin.h>  // AVX指令集
#include <stdlib.h>     // aligned_alloc
#include <omp.h>  // 添加OpenMP头文件

// 检测CPU支持的指令集
#ifdef __AVX512F__
    #define USE_AVX512
#endif

#ifdef USE_AVX512
    #define VECTOR_SIZE 8
    typedef __m512d vector_d;
    #define vector_load _mm512_loadu_pd
    #define vector_store _mm512_storeu_pd
    #define vector_set1 _mm512_set1_pd
    #define vector_add _mm512_add_pd
    #define vector_mul _mm512_mul_pd
    #define vector_fmadd _mm512_fmadd_pd
    #define vector_setzero _mm512_setzero_pd
#else
    #define VECTOR_SIZE 4
    typedef __m256d vector_d;
    #define vector_load _mm256_loadu_pd
    #define vector_store _mm256_storeu_pd
    #define vector_set1 _mm256_set1_pd
    #define vector_add _mm256_add_pd
    #define vector_mul _mm256_mul_pd
    #define vector_fmadd _mm256_fmadd_pd
    #define vector_setzero _mm256_setzero_pd
#endif

// 转置块矩阵
static void transpose_block(double *dst, const double *src, 
                          int n, int m, int ld_src) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            dst[j*n + i] = src[i + j*ld_src];
}

/* 为小矩阵使用简单实现 */
static void do_block_small(int M, int N, int K,
                         double *a, int lda, int start_a_i, int start_a_j,
                         double *b, int ldb, int start_b_i, int start_b_j,
                         double *c, int ldc, int start_c_i, int start_c_j)
{
    // 使用2x2的小块计算，减少寄存器压力
    for (int i = 0; i < M; i += 2) {
        for (int j = 0; j < N; j += 2) {
            // 使用寄存器存储中间结果
            double c00 = C(start_c_i + i, start_c_j + j);
            double c01 = C(start_c_i + i, start_c_j + j + 1);
            double c10 = C(start_c_i + i + 1, start_c_j + j);
            double c11 = C(start_c_i + i + 1, start_c_j + j + 1);
            
            // 计算2x2块
            for (int k = 0; k < K; k++) {
                double a0 = A(start_a_i + i, start_a_j + k);
                double a1 = A(start_a_i + i + 1, start_a_j + k);
                
                double b0 = B(start_b_i + k, start_b_j + j);
                double b1 = B(start_b_i + k, start_b_j + j + 1);
                
                // 标量乘加操作
                c00 += a0 * b0;  c01 += a0 * b1;
                c10 += a1 * b0;  c11 += a1 * b1;
            }
            
            // 写回结果
            C(start_c_i + i, start_c_j + j) = c00;
            C(start_c_i + i, start_c_j + j + 1) = c01;
            C(start_c_i + i + 1, start_c_j + j) = c10;
            C(start_c_i + i + 1, start_c_j + j + 1) = c11;
        }
    }
    
    // 处理边界情况
    if (M % 2 != 0) {
        int i = M - 1;
        for (int j = 0; j < N; j++) {
            double cij = C(start_c_i + i, start_c_j + j);
            for (int k = 0; k < K; k++) {
                cij += A(start_a_i + i, start_a_j + k) * 
                      B(start_b_i + k, start_b_j + j);
            }
            C(start_c_i + i, start_c_j + j) = cij;
        }
    }
    
    if (N % 2 != 0) {
        int j = N - 1;
        for (int i = 0; i < M - 1; i++) {
            double cij = C(start_c_i + i, start_c_j + j);
            for (int k = 0; k < K; k++) {
                cij += A(start_a_i + i, start_a_j + k) * 
                      B(start_b_i + k, start_b_j + j);
            }
            C(start_c_i + i, start_c_j + j) = cij;
        }
    }
}

/* 大矩阵的块计算 */
static void do_block(int M, int N, int K,
                    double *a, int lda, int start_a_i, int start_a_j,
                    double *b, int ldb, int start_b_i, int start_b_j,
                    double *c, int ldc, int start_c_i, int start_c_j)
{
    // 每个线程使用自己的缓存块
    #pragma omp threadprivate(a_block, b_block)
    static double a_block[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(64)));
    static double b_block[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(64)));
    
    // 预加载数据 - 可以并行
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            // 预加载A块
            for (int i = 0; i < M; i++) {
                for (int k = 0; k < K; k++) {
                    a_block[k*M + i] = A(start_a_i + i, start_a_j + k);
                }
            }
        }
        
        #pragma omp section
        {
            // 预加载B块
            for (int k = 0; k < K; k++) {
                for (int j = 0; j < N; j++) {
                    b_block[k*N + j] = B(start_b_i + k, start_b_j + j);
                }
            }
        }
    }
    
    // 主计算循环 - 并行处理行
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; i += 6) {
        vector_d c_local[6][2];  // 线程私有的累加器
        
        for (int j = 0; j < N; j += 8) {
            // 初始化累加器
            for (int ii = 0; ii < 6; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    c_local[ii][jj] = vector_setzero();
                }
            }
            
            // 计算6x8块
            for (int k = 0; k < K; k++) {
                // 预取下一个k的数据
                if (k + 4 < K) {
                    _mm_prefetch(&a_block[(k+4)*M + i], _MM_HINT_T0);
                    _mm_prefetch(&b_block[(k+4)*N + j], _MM_HINT_T1);
                }
                
                vector_d b0 = vector_load(&b_block[k*N + j]);
                vector_d b1 = vector_load(&b_block[k*N + j + 4]);
                
                for (int ii = 0; ii < 6 && i + ii < M; ii++) {
                    vector_d a0 = vector_set1(a_block[k*M + i + ii]);
                    c_local[ii][0] = vector_fmadd(a0, b0, c_local[ii][0]);
                    c_local[ii][1] = vector_fmadd(a0, b1, c_local[ii][1]);
                }
            }
            
            // 写回结果 - 需要原子操作
            #pragma omp critical
            {
                for (int ii = 0; ii < 6 && i + ii < M; ii++) {
                    if (j + 4 <= N) {
                        vector_store(&C(start_c_i + i + ii, start_c_j + j),
                                   vector_add(c_local[ii][0], 
                                            vector_load(&C(start_c_i + i + ii, start_c_j + j))));
                    }
                    if (j + 8 <= N) {
                        vector_store(&C(start_c_i + i + ii, start_c_j + j + 4),
                                   vector_add(c_local[ii][1], 
                                            vector_load(&C(start_c_i + i + ii, start_c_j + j + 4))));
                    }
                }
            }
        }
    }
}

/* Routine for computing C = A * B + C */
void MY_MMult(int m, int n, int k, double *a, int lda,
              double *b, int ldb, double *c, int ldc)
{
    // 设置线程数
    int num_threads = omp_get_num_procs();  // 获取CPU核心数
    omp_set_num_threads(num_threads);
    
    // 对小矩阵使用简单实现
    if (m <= 64 || n <= 64 || k <= 64) {
        do_block_small(m, n, k, a, lda, 0, 0, b, ldb, 0, 0, c, ldc, 0, 0);
        return;
    }
    
    // 大矩阵使用并行实现
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < m; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            for (int p = 0; p < k; p += BLOCK_SIZE) {
                int M = min(BLOCK_SIZE, m - i);
                int N = min(BLOCK_SIZE, n - j);
                int K = min(BLOCK_SIZE, k - p);
                
                do_block(M, N, K,
                        a, lda, i, p,
                        b, ldb, p, j,
                        c, ldc, i, j);
            }
        }
    }
} 