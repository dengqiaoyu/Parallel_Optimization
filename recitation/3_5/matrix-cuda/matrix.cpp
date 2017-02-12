#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "matrix.h"
#include "CycleTimer.h"

#define MATRIX 1
/* Error Tolerance */
float errTolerance = 0.001;
/* Limit to errors before giving up */
int errLimit = 10;
/* Optimum number of operations for measurements */
/* Want around 100ms for 1GF */
long targetOps = 100000000;

/* Preallocated blocks */
static int allocN = -1;
static float *aData = NULL;
static float *bData = NULL;
static float *tData = NULL;
static float *gData = NULL;
static float *sData = NULL;

/* Forward pointers */
void multMatrixSimple(int N, float *matA, float *matB, float *matC);

void setup(int N) {
    if (allocN == N)
	return;
    if (allocN > 0) {
	free(sData);
	free(aData);
	free(bData);
	free(tData);
	free(gData);
    }
    if (N > 0) {
	aData = (float *) calloc(N*N, sizeof(float));
	bData = (float *) calloc(N*N, sizeof(float));
	tData = (float *) calloc(N*N, sizeof(float));
	gData = (float *) calloc(N*N, sizeof(float));
	sData = (float *) calloc(N*N, sizeof(float));
	/* Generate random data for A & B */
	for (int i = 0; i < N; i++)
	    for (int j = 0; j < N; j++) {
		// Random numbers between 1.0 and 10.0
		aData[RM(i,j,N)] = 1.0 + ((float) rand()/RAND_MAX) * 9.0;
		bData[RM(i,j,N)] = 1.0 + ((float) rand()/RAND_MAX) * 9.0;
	    }
    }
    /* Generate reference data */
    multMatrixSimple(N, aData, bData, gData);
    allocN = N;
    // Get things ready on the Cuda side
    cudaSetup(N, aData, bData, gData);
}


// Get scratch for matrix
static float *scratchMatrix(int N) {
    if (allocN != N) {
	setup(N);
    }
    return sData;
}

// Test two matrices for equality.  Return number of mismatches
int checkMatrix(int N, float *matTest, float *matGood) {
    int errCount = 0;
    for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++) {
	    int idx = RM(i,j,N);
	    float test = matTest[idx];
	    float good = matGood[idx];
	    float err = (test-good)/test;
	    if (err < -errTolerance || err > errTolerance) {
		if (++errCount <= errLimit)
		    fprintf(stderr,
"\tMismatch.  N=%d.\ttest[%d][%d] = %.3f.  good[%d][%d] = %.3f\n",
			    N, i, j, test, i, j, good);
	    }
	    
	}
    return errCount;
}

// Transpose a matrix
void transposeMatrix(int N, float *matS, float *matD) {
    for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++)
	    matD[CM(i,j,N)] = matS[RM(i,j,N)];
}

// Standard multiplication
void multMatrixSimple(int N, float *matA, float *matB, float *matC) {
    for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++) {
	    float sum = 0.0;	    
	    for (int k = 0; k < N; k++)
		sum += matA[RM(i,k,N)] * matB[RM(k,j,N)];
	    matC[RM(i,j,N)] = sum;
	}
}

// Multiplication, first transposing B 
void multMatrixTransposed(int N, float *matA, float *matB, float *matC) {
    float *tranB = scratchMatrix(N);
    transposeMatrix(N, matB, tranB);
    for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++) {
	    float sum = 0.0;	    
	    for (int k = 0; k < N; k++)
		sum += matA[RM(i,k,N)] * tranB[RM(j,k,N)];
	    matC[RM(i,j,N)] = sum;
	}
}

void multMatrixBlocked(int N, float *matA, float *matB, float *matC) {
    /* Zero out C */
    memset(matC, 0, N * N * sizeof(float));
    int i, j, k;
    for (i = 0; i <= N-SBLK; i+= SBLK) {
	for (j = 0; j <= N-SBLK; j+= SBLK) {
	    for (k = 0; k <= N-SBLK; k+=SBLK) {
		for (int bi = 0; bi < SBLK; bi++) 
		    for (int bj = 0; bj < SBLK; bj++) {
			float sum = 0.0;
			for (int bk =0; bk < SBLK; bk++)
			    sum += matA[RM(i+bi,k+bk,N)]
				* matB[RM(k+bk,j+bj,N)];
			matC[RM(i+bi,j+bj,N)] += sum;
		    }
	    }
	    // Finish rest of k
	    for (int bi = 0; bi < SBLK; bi++) 
		for (int bj = 0; bj < SBLK; bj++) {
		    float sum = 0.0;
		    for (int rk = k; rk < N; rk++)
			sum += matA[RM(i+bi,rk,N)] * matB[RM(rk,j+bj,N)];
		    matC[RM(i+bi,j+bj,N)] += sum;
		}
	}
	// Finish rest of j
	for (int bi = 0; bi < SBLK; bi++)
	    for (int rj = j; rj < N; rj++) {
		float sum = 0.0;	    
		for (k = 0; k < N; k++)
		    sum += matA[RM(i+bi,k,N)] * matB[RM(k,rj,N)];
		matC[RM(i+bi,rj,N)] += sum;
	    }
    }
    // Finish rest of i
    for (int ri = i; ri < N; ri++)
	for (j = 0; j < N; j++) {
	    float sum = 0.0;	    
	    for (k = 0; k < N; k++)
		sum += matA[RM(ri,k,N)] * matB[RM(k,j,N)];
	    matC[RM(ri,j,N)] += sum;
	}
}

void multMatrixTransposeBlocked(int N, float *matA, float *matB, float *matC) {
    float *tranB = scratchMatrix(N);
    transposeMatrix(N, matB, tranB);
    /* Zero out C */
    memset(matC, 0, N * N * sizeof(float));
    int i, j, k;
    for (i = 0; i <= N-SBLK; i+= SBLK) {
	for (j = 0; j <= N-SBLK; j+= SBLK) {
	    for (k = 0; k <= N-SBLK; k+=SBLK) {
		for (int bi = 0; bi < SBLK; bi++) 
		    for (int bj = 0; bj < SBLK; bj++) {
			float sum = 0.0;
			for (int bk =0; bk < SBLK; bk++)
			    sum += matA[RM(i+bi,k+bk,N)]
				* tranB[RM(j+bj,k+bk,N)];
			matC[RM(i+bi,j+bj,N)] += sum;
		    }
	    }
	    // Finish rest of k
	    for (int bi = 0; bi < SBLK; bi++) 
		for (int bj = 0; bj < SBLK; bj++) {
		    float sum = 0.0;
		    for (int rk = k; rk < N; rk++)
			sum += matA[RM(i+bi,rk,N)] * tranB[RM(j+bj,rk,N)];
		    matC[RM(i+bi,j+bj,N)] += sum;
		}
	}
	// Finish rest of j
	for (int bi = 0; bi < SBLK; bi++)
	    for (int rj = j; rj < N; rj++) {
		float sum = 0.0;	    
		for (k = 0; k < N; k++)
		    sum += matA[RM(i+bi,k,N)] * tranB[RM(rj,k,N)];
		matC[RM(i+bi,rj,N)] += sum;
	    }
    }
    // Finish rest of i
    for (int ri = i; ri < N; ri++)
	for (j = 0; j < N; j++) {
	    float sum = 0.0;	    
	    for (k = 0; k < N; k++)
		sum += matA[RM(ri,k,N)] * tranB[RM(j,k,N)];
	    matC[RM(ri,j,N)] += sum;
	}
}

void multMatrixFastBlocked(int N, float *matA, float *matB, float *matC) {
    float *tranB = scratchMatrix(N);
    transposeMatrix(N, matB, tranB);
    /* Zero out C */
    memset(matC, 0, N * N * sizeof(float));
    int i, j, k;
    for (i = 0; i <= N-SBLK; i+= SBLK) {
	for (j = 0; j <= N-SBLK; j+= SBLK) {
	    for (k = 0; k <= N-SBLK; k+=SBLK) {
		for (int bi = 0; bi < SBLK; bi++) 
		    for (int bj = 0; bj < SBLK; bj++) {
			float sum = 0.0;
			int aIdx = RM(i+bi,k,N);
			int bIdx = RM(j+bj,k,N);
			float p0 = matA[aIdx+0] * tranB[bIdx+0];
			float p1 = matA[aIdx+1] * tranB[bIdx+1];
			float s01 = p0+p1;
			float p2 = matA[aIdx+2] * tranB[bIdx+2];
			float p3 = matA[aIdx+3] * tranB[bIdx+3];
			float s23 = p2+p3;
			float s0123 = s01+s23;
			float p4 = matA[aIdx+4] * tranB[bIdx+4];
			float p5 = matA[aIdx+5] * tranB[bIdx+5];
			float s45 = p4+p5;
			float p6 = matA[aIdx+6] * tranB[bIdx+6];
			float p7 = matA[aIdx+7] * tranB[bIdx+7];
			float s67 = p6+p7;
			float s4567 = s45+s67;
			sum += (s0123+s4567);
			matC[RM(i+bi,j+bj,N)] += sum;
		    }
	    }
	    // Finish rest of k
	    for (int bi = 0; bi < SBLK; bi++) 
		for (int bj = 0; bj < SBLK; bj++) {
		    float sum = 0.0;
		    for (int rk = k; rk < N; rk++)
			sum += matA[RM(i+bi,rk,N)] * tranB[RM(j+bj,rk,N)];
		    matC[RM(i+bi,j+bj,N)] += sum;
		}
	}
	// Finish rest of j
	for (int bi = 0; bi < SBLK; bi++)
	    for (int rj = j; rj < N; rj++) {
		float sum = 0.0;	    
		for (k = 0; k < N; k++)
		    sum += matA[RM(i+bi,k,N)] * tranB[RM(rj,k,N)];
		matC[RM(i+bi,rj,N)] += sum;
	    }
    }
    // Finish rest of i
    for (int ri = i; ri < N; ri++)
	for (j = 0; j < N; j++) {
	    float sum = 0.0;	    
	    for (k = 0; k < N; k++)
		sum += matA[RM(ri,k,N)] * tranB[RM(j,k,N)];
	    matC[RM(ri,j,N)] += sum;
	}
}

static int runMM(int N, mmul_t method) {
    switch (method) {
    case MMUL_REFERENCE:
	multMatrixSimple(N, aData, bData, tData);
	break;
    case MMUL_TRANSPOSE:
	multMatrixTransposed(N, aData, bData, tData);
	break;
    case MMUL_BLK:
	multMatrixBlocked(N, aData, bData, tData);
	break;
    case MMUL_TRANSPOSE_BLK:
	multMatrixTransposeBlocked(N, aData, bData, tData);
	break;
    case MMUL_FAST_BLK:
	multMatrixFastBlocked(N, aData, bData, tData);
	break;
    default:
	fprintf(stderr, "Haven't implemented method yet\n");
	return 0;
    }
    return 1;
}

double benchMM(int N, mmul_t method) {
    setup(N);
    if (!runMM(N, method))
	return 1000.0;
    if (checkMatrix(N, tData, gData) > 0)
	return 1000.0;
    /* Now do the real benchmarking */
    long ops = (long) 2 * N * N * N;
    long runs = (targetOps+ops-1)/ops;
    double startTime = CycleTimer::currentSeconds();
    for (long r = 0; r < runs; r++)
	runMM(N, method);
    double endTime = CycleTimer::currentSeconds();
    double ms = (endTime - startTime) * 1000.0;
    double gflops = (long) (runs*ops)/ms * 1e-6;
    fprintf(stderr,
	    "%ld runs, %ld ops/run, %.2f ms, %.3f GFlops\n",
	    runs, ops, ms, gflops);
    return gflops;
}
