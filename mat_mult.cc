#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <chrono>
#if defined(USE_MKL)
#include "mkl.h"
#elif defined(USE_OPENBLAS)
#include "cblas.h"
#endif

using timer_clock= std::chrono::high_resolution_clock;


inline unsigned int min(unsigned int a, unsigned int b)
{
  return (a < b) ? a : b;
}


void naive_multiply(double **matA, double **matB, double **matC, unsigned int n,
		    unsigned int m, unsigned int l)
{
  timer_clock::time_point t1= timer_clock::now();
  unsigned int i, j, k;
  // Perform the matrix-matrix multiplication naively
  for (i= 0; i < n; ++i){
    for (j= 0; j < l; ++j){
      for (k= 0; k < m; ++k){
	matC[i][j] += matA[i][k] * matB[k][j];
      }
    }
  }
  timer_clock::time_point t2= timer_clock::now();
  std::chrono::duration<double> time_span= std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  printf("Naive multiply took: %.3lf seconds\n", time_span.count());
}

void block_transpose_multiply(double **matA, double **matBT, double **matC, unsigned int n,
                              unsigned int m, unsigned int l, unsigned int blockSize)
{
  timer_clock::time_point t1= timer_clock::now();
  unsigned int i, j, jj, k, kk;
  // Perform the matrix-matrix multiplication with a bit of blocking and
  // loop unrolling. We cheat with the loop unrolling, and just expect the
  // dimensions that are passed in to be a multiple of two.
  for (kk= 0; kk < m; kk+= blockSize){
    for(jj= 0; jj < l; jj+= blockSize){
      for(i= 0; i < n; i+= 2){
	for(j= jj; j < min(l, jj + blockSize); j+= 2){
	  for(k =kk; k < min(m, kk + blockSize); ++k){
	    matC[i][j] += matA[i][k] * matBT[j][k];
	    matC[i][j+1] += matA[i][k] * matBT[j+1][k];
	    matC[i+1][j] += matA[i+1][k] * matBT[j][k];
	    matC[i+1][j+1] += matA[i+1][k] * matBT[j+1][k];
	  }
	}
      }
    }
  }
  timer_clock::time_point t2= timer_clock::now();
  std::chrono::duration<double> time_span= std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  printf("Transpose multiply took: %.3lf seconds\n", time_span.count());
}


int main(int argc, char** argv)
{
  double *dataA= NULL;
  double *dataB= NULL;
  double *dataC= NULL;
  double **matA= NULL;
  double **matB= NULL;
  double **matC= NULL;

  // Generally our inputs are A = n x m matrix, B = m x l matrix, C = n x l matrix
  // But we'll keep it simple and set n = m  = l

  unsigned int n, m, l;

  unsigned int i, j, k;

  unsigned int blockSize;

#if defined(USE_BLOCK_BT)
  if (argc < 3){
    printf("Usage: %s n blockSize\n", argv[0]);
    return 1;
  }
#else
  if (argc < 2){
    printf("Usage: %s n\n", argv[0]);
    return 1;
  }
#endif

  n= (unsigned int)atoi(argv[1]);
  m= n;
  l= n;

#if defined(USE_BLOCK_BT)
  blockSize= (unsigned int)atoi(argv[2]);
#endif

  matA= (double**)malloc(n*sizeof(double*));
  matB= (double**)malloc(m*sizeof(double*));
  matC= (double**)malloc(n*sizeof(double*));

  // Assign memory
#if defined(USE_NAIVE)
  timer_clock::time_point t1= timer_clock::now();

  for(i=0; i < n; ++i){
    matA[i]= (double*)malloc(m*sizeof(double*));
    matC[i]= (double*)malloc(l*sizeof(double*));
  }

  for(i= 0; i < m; ++i){
    matB[i]= (double*)malloc(l*sizeof(double*));
  }

#else
  dataA= (double*)malloc(n*m*sizeof(double));
  dataB= (double*)malloc(m*l*sizeof(double));
  dataC= (double*)malloc(m*l*sizeof(double));

  timer_clock::time_point t1= timer_clock::now();
  // Set up the matrices in row major format
  for(i= 0; i < n; ++i){
    matA[i]= dataA + i*m;
    matC[i]= dataC + i*l;
  }
#if defined(USE_BLOCK_BT)
  for(i= 0; i < l; ++i){
    matB[i]= dataB + i*m;
  }
#else
  for(i= 0; i < m; ++i){
    matB[i]= dataB + i*l;
  }
#endif
#endif

  // Initialise matrices. Matrices A and B get random data. Results matrix C
  // is initialised to zero
  srand(time(NULL));
  for (i= 0; i < n; ++i){
    for (j= 0; j < m; ++j){
      matA[i][j]= ((double)rand()) / RAND_MAX;
    }

    for (j= 0; j < l; ++j){
      matC[i][j]= 0.0;
    }
  }

#if defined(USE_BLOCK_BT)
  for (i= 0; i < l; ++i){
    for (j= 0; j < m; ++j){
      matB[i][j]= ((double)rand()) / RAND_MAX;
    }
  }
#else
  for (i= 0; i < m; ++i){
    for (j= 0; j < l; ++j){
      matB[i][j]= ((double)rand()) / RAND_MAX;
    }
  }
#endif

  timer_clock::time_point t2= timer_clock::now();
  std::chrono::duration<double> time_span= std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  printf("Set up of matrices took: %.3lf seconds\n", time_span.count());

  // Perform the matrix-matrix multiplication naively
  printf("Performing multiply\n");
#if defined(USE_NAIVE) || defined(USE_ONEDARRAY)
  naive_multiply(matA, matB, matC, n, m, l);
#elif defined(USE_BLOCK_BT)
  block_transpose_multiply(matA, matB, matC, n, m, l, blockSize);
#elif defined(USE_TRANSPOSE)
#elif defined(USE_OPENBLAS) || defined(USE_MKL)
  t1= timer_clock::now();
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, m, l, 1, dataA,
	      m, dataB, l, 1, dataC, m);
  t2= timer_clock::now();
  time_span= std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  printf("Transpose multiply took: %.3lf seconds\n", time_span.count());
#endif

  // Free memory
  free(matA);
  free(dataA);
  free(matB);
  free(dataB);
  free(matC);
  free(dataC);
}
