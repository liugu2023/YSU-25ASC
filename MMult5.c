#include <immintrin.h>
#include <stdlib.h>
#include <omp.h>

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]
#define min(i, j) ((i) < (j) ? (i) : (j))

// 调整分块大小以适应200
#define BLOCK_SIZE 50        // 200 = 50 * 4，减少边界处理
#define SMALL_BLOCK_SIZE 10  // 50 = 10 * 5
#define MICRO_BLOCK_SIZE 5   // 更适合于非2的幂次数据规模
#define VECTOR_SIZE 4        // AVX2 = 4 doubles
#define UNROLL_K 4          // K维度的展开因子

typedef __m256d vector_d;
#define vector_load _mm256_loadu_pd
#define vector_store _mm256_storeu_pd
#define vector_set1 _mm256_set1_pd
#define vector_add _mm256_add_pd
#define vector_mul _mm256_mul_pd
#define vector_fmadd _mm256_fmadd_pd
#define vector_setzero _mm256_setzero_pd

// 添加新的数据结构来存储重排后的矩阵块
struct matrix_block {
    double *data;
    int rows;
    int cols;
    int stride;
};

// 创建并初始化矩阵块
static inline struct matrix_block create_block(int rows, int cols) {
    struct matrix_block block;
    block.data = aligned_alloc(64, rows * cols * sizeof(double));
    block.rows = rows;
    block.cols = cols;
    block.stride = (cols + VECTOR_SIZE - 1) & ~(VECTOR_SIZE - 1); // 对齐到向量大小
    return block;
}

// 释放矩阵块
static inline void destroy_block(struct matrix_block *block) {
    free(block->data);
}

/* 计算微块 */
static inline void compute_micro_kernel(int K,
                                      const double *a_block, int a_stride,
                                      const double *b_block, int b_stride,
                                      double *c_local) 
{
    // 使用8个向量寄存器存储中间结果
    vector_d c00 = vector_setzero();
    vector_d c10 = vector_setzero();
    vector_d c20 = vector_setzero();
    vector_d c30 = vector_setzero();
    vector_d c40 = vector_setzero();
    
    // 手动展开K循环，提高指令级并行
    for (int k = 0; k < K - 3; k += 4) {
        // 预取下一轮数据
        _mm_prefetch(&a_block[(k + 4) * a_stride], _MM_HINT_T0);
        _mm_prefetch(&b_block[(k + 4) * b_stride], _MM_HINT_T1);
        
        // 加载4个连续的B向量
        vector_d b0 = vector_load(&b_block[k * b_stride]);
        vector_d b1 = vector_load(&b_block[(k+1) * b_stride]);
        vector_d b2 = vector_load(&b_block[(k+2) * b_stride]);
        vector_d b3 = vector_load(&b_block[(k+3) * b_stride]);
        
        // 计算第一行
        vector_d a0 = vector_set1(a_block[k * a_stride]);
        c00 = vector_fmadd(a0, b0, c00);
        a0 = vector_set1(a_block[(k+1) * a_stride]);
        c00 = vector_fmadd(a0, b1, c00);
        a0 = vector_set1(a_block[(k+2) * a_stride]);
        c00 = vector_fmadd(a0, b2, c00);
        a0 = vector_set1(a_block[(k+3) * a_stride]);
        c00 = vector_fmadd(a0, b3, c00);
        
        // 计算其他行（类似模式）
        // ... 重复上述模式直到第5行
    }
    
    // 处理剩余的K
    for (int k = K - (K % 4); k < K; k++) {
        vector_d b0 = vector_load(&b_block[k * b_stride]);
        for (int i = 0; i < 5; i++) {
            vector_d a0 = vector_set1(a_block[k * a_stride + i]);
            c[i] = vector_fmadd(a0, b0, c[i]);
        }
    }
    
    // 存储结果
    vector_store(&c_local[0], c00);
    vector_store(&c_local[4], c10);
    vector_store(&c_local[8], c20);
    vector_store(&c_local[12], c30);
    vector_store(&c_local[16], c40);
}

// 优化的矩阵打包函数
static void pack_matrix_A(struct matrix_block *dest,
                         const double *src, int lda,
                         int start_i, int start_j,
                         int M, int K) 
{
    // 使用向量化加载提高带宽利用率
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i += SMALL_BLOCK_SIZE) {
        for (int k = 0; k < K; k += SMALL_BLOCK_SIZE) {
            // 一次加载多行
            for (int ii = 0; ii < SMALL_BLOCK_SIZE && i + ii < M; ii += VECTOR_SIZE) {
                for (int kk = 0; kk < SMALL_BLOCK_SIZE && k + kk < K; kk++) {
                    vector_d tmp = vector_load(&src[(start_j + k + kk)*lda + (start_i + i + ii)]);
                    vector_store(&dest->data[(k+kk)*dest->stride + (i+ii)], tmp);
                }
            }
        }
    }
}

static void pack_matrix_B(struct matrix_block *dest,
                         const double *src, int ldb,
                         int start_i, int start_j,
                         int K, int N) 
{
    // 使用更大的打包块，减少打包开销
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < K; k += SMALL_BLOCK_SIZE) {
        for (int j = 0; j < N; j += SMALL_BLOCK_SIZE) {
            // 一次打包更大的块
            for (int kk = 0; kk < SMALL_BLOCK_SIZE && k + kk < K; kk++) {
                for (int jj = 0; jj < SMALL_BLOCK_SIZE && j + jj < N; jj++) {
                    dest->data[k*dest->stride + j + jj] = 
                        src[(start_j + j + jj)*ldb + (start_i + k + kk)];
                }
            }
        }
    }
}

/* 大矩阵的块计算 */
static void do_block(int M, int N, int K,
                    double *a, int lda, int start_a_i, int start_a_j,
                    double *b, int ldb, int start_b_i, int start_b_j,
                    double *c, int ldc, int start_c_i, int start_c_j)
{
    // 使用更大的缓存对齐
    static double a_block[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(64)));
    static double b_block[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(64)));
    
    // 使用SIMD加载数据
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (int i = 0; i < M; i += VECTOR_SIZE) {
                for (int k = 0; k < K; k++) {
                    vector_d tmp = vector_load(&A(start_a_i + i, start_a_j + k));
                    vector_store(&a_block[k*M + i], tmp);
                }
            }
        }
        
        #pragma omp section
        {
            for (int k = 0; k < K; k++) {
                for (int j = 0; j < N; j += VECTOR_SIZE) {
                    vector_d tmp = vector_load(&B(start_b_i + k, start_b_j + j));
                    vector_store(&b_block[k*N + j], tmp);
                }
            }
        }
    }
    
    // 3. 优化并行计算策略
    #pragma omp parallel
    {
        // 每个线程使用私有的累加缓冲区
        double c_block[MICRO_BLOCK_SIZE * VECTOR_SIZE] __attribute__((aligned(32)));
        
        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < M; i += MICRO_BLOCK_SIZE) {
            for (int j = 0; j < N; j += VECTOR_SIZE) {
                // 初始化累加器
                memset(c_block, 0, sizeof(c_block));
                
                // 计算微块
                for (int k = 0; k < K; k += MICRO_BLOCK_SIZE) {
                    compute_micro_kernel(min(MICRO_BLOCK_SIZE, K-k),
                                      &a_block[k*M + i], M,
                                      &b_block[k*N + j], N,
                                      c_block);
                }
                
                // 使用SIMD更新结果
                #pragma omp critical
                {
                    for (int ii = 0; ii < MICRO_BLOCK_SIZE && i + ii < M; ii++) {
                        vector_d sum = vector_load(&c_block[ii * VECTOR_SIZE]);
                        vector_d old = vector_load(&C(start_c_i + i + ii, start_c_j + j));
                        vector_store(&C(start_c_i + i + ii, start_c_j + j),
                                   vector_add(sum, old));
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
    int num_threads = omp_get_num_procs();
    omp_set_num_threads(num_threads);

    // 大矩阵使用分块并行实现
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