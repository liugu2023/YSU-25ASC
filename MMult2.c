/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Create macro to let X( i ) equal the ith element of x */
#define X(i) x[ (i)*incx ]

#define LDA (M+4)    // 当 M = p 时，lda = p+4
#define LDB (K+4)    // 当 K = p 时，ldb = p+4
#define LDC (M+4)    // 当 M = p 时，ldc = p+4

void AddDot( int k, double *x, int incx,  double *y, double *gamma )
{
  double temp = 0.0;
  int p;

  for ( p=0; p<k-3; p+=4 ){
    temp += X(p) * y[p] + 
            X(p+1) * y[p+1] + 
            X(p+2) * y[p+2] + 
            X(p+3) * y[p+3];
  }

  for (; p<k; p++){
    temp += X(p) * y[p];
  }

  *gamma += temp;
}

/* Routine for computing C = A * B + C */
void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j;

  if (m % 320 == 0) {
    lda = m + 4;  // 添加4个元素的padding
  } else {
    lda = m;
  }

  for ( j=0; j<n; j+=1 ){        /* Loop over the columns of C */
    for ( i=0; i<m; i+=1 ){        /* Loop over the rows of C */
      /* Update the C( i,j ) with the inner product of the ith row of A
         and the jth column of B */
      AddDot( k, &A( i,0 ), lda, &B( 0,j ), &C( i,j ) );
    }
  }
} 