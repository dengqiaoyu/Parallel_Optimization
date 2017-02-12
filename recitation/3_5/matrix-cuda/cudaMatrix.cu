#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "CycleTimer.h"

#include "matrix.h"

// Integer division, rounding up
static inline int updiv(int n, int d) {
    return (n+d-1)/d;
}

/* Transpose matrix */
__global__ void
cudaTransposeKernel(int N, const float  *dmatS, float *dmatD) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N)
	return;
    dmatD[CM(i,j,N)] = dmatS[RM(i,j,N)];
}

__global__ void
cudaSimpleKernelOld(int N, float *dmatA, float *dmatB, float *dmatC) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N)
	return;
    float sum = 0.0;
    for (int k = 0; k < N; k++) {
	sum += dmatA[RM(i,k,N)] * dmatB[RM(k,j,N)];
    }
    dmatC[RM(i,j,N)] = sum;
}

__global__ void
cudaSimpleKernel(int N, float*  dmatA, float* dmatB, float * dmatC) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N)
	return;
    float sum = 0.0;
    for (int k = 0; k < N; k++) {
	sum += dmatA[RM(i,k,N)] * dmatB[RM(k,j,N)];
    }
    dmatC[RM(i,j,N)] = sum;
}

__global__ void
cudaTransposedKernel(int N, float *dmatA, float *dmatB, float *dmatC) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N)
	return;
    float sum = 0.0;
    for (int k = 0; k < N; k++) {
	sum += dmatA[RM(i,k,N)] * dmatB[CM(k,j,N)];
    }
    dmatC[RM(i,j,N)] = sum;
}

__global__ void
cudaBlockKernelOld(int N, float *dmatA, float *dmatB, float *dmatC) {
    // Assume that thread block contains submatrix of size LBLK x LBLK
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int bi = threadIdx.x;
    int bj = threadIdx.y;

    float sum = 0.0; // Accumulate result for C[i][j]

    // Shared space for two submatrices of A and B
    __shared__ float subA[LBLK*LBLK];
    __shared__ float subB[LBLK*LBLK];

    // Loop over k to compute product of all submatrices A[i][k] and B[k][j]
    for (int k = 0; k < N; k+= LBLK) {
	// Grab the two submatrices
	if (i < N && k+bj < N)
	    subA[RM(bi,bj,LBLK)] = dmatA[RM(i,k+bj,N)];
	else
	    subA[RM(bi,bj,LBLK)] = 0.0;

	if (j < N && k+bi < N)
	    subB[RM(bi,bj,LBLK)] = dmatB[RM(k+bi,j,N)];
	else
	    subB[RM(bi,bj,LBLK)] = 0.0;

	// Wait until entire block gets filled
	__syncthreads();

	// Generate contribution to C[i][j] of these submatrices
	for (int bk = 0; bk < LBLK; bk++)
	    sum += subA[RM(bi,bk,LBLK)] * subB[RM(bk,bj,LBLK)];
	// Wait until all products computed
	__syncthreads();
    }
    if (i < N && j < N)
	dmatC[RM(i,j,N)] = sum;
}

__global__ void
cudaBlockKernel(int N, float *dmatA, float *dmatB, float *dmatC) {
    // Assume that thread block contains submatrix of size LBLK x LBLK
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int bi = threadIdx.y;
    int bj = threadIdx.x;

    float sum = 0.0; // Accumulate result for C[i][j]

    // Shared space for two submatrices of A and B
    __shared__ float subA[LBLK*LBLK];
    __shared__ float subB[LBLK*LBLK];

    // Loop over k to compute product of all submatrices A[i][k] and B[k][j]
    for (int k = 0; k < N; k+= LBLK) {
	// Grab the two submatrices
	if (i < N && k+bj < N)
	    subA[RM(bi,bj,LBLK)] = dmatA[RM(i,k+bj,N)];
	else
	    subA[RM(bi,bj,LBLK)] = 0.0;

	if (j < N && k+bi < N)
	    subB[RM(bi,bj,LBLK)] = dmatB[RM(k+bi,j,N)];
	else
	    subB[RM(bi,bj,LBLK)] = 0.0;

	// Wait until entire block gets filled
	__syncthreads();

	// Generate contribution to C[i][j] of these submatrices
	for (int bk = 0; bk < LBLK; bk++)
	    sum += subA[RM(bi,bk,LBLK)] * subB[RM(bk,bj,LBLK)];

	// Wait until all products computed
	__syncthreads();
    }
    if (i < N && j < N)
	dmatC[RM(i,j,N)] = sum;
}

// Transpose submatrix of B as read it in.  Decreases performance.
__global__ void
cudaBlockTransposeKernel(int N, float *dmatA, float *dmatB, float *dmatC) {
    // Assume that thread block contains submatrix of size LBLK x LBLK
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int bi = threadIdx.y;
    int bj = threadIdx.x;

    float sum = 0.0; // Accumulate result for C[i][j]

    // Shared space for two submatrices of A and B
    __shared__ float subA[LBLK*LBLK];
    __shared__ float subB[LBLK*LBLK];

    // Loop over k to compute product of all submatrices A[i][k] and B[k][j]
    for (int k = 0; k < N; k+= LBLK) {
	// Grab the two submatrices
	if (i < N && k+bj < N)
	    subA[RM(bi,bj,LBLK)] = dmatA[RM(i,k+bj,N)];
	else
	    subA[RM(bi,bj,LBLK)] = 0.0;

	if (j < N && k+bi < N)
	    subB[CM(bi,bj,LBLK)] = dmatB[RM(k+bi,j,N)];
	else
	    subB[CM(bi,bj,LBLK)] = 0.0;

	// Wait until entire block gets filled
	__syncthreads();

	// Generate contribution to C[i][j] of these submatrices
	for (int bk = 0; bk < LBLK; bk++)
	    sum += subA[RM(bi,bk,LBLK)] * subB[CM(bk,bj,LBLK)];

	// Wait until all products computed
	__syncthreads();
    }
    if (i < N && j < N)
	dmatC[RM(i,j,N)] = sum;
}




// The following version only works when N is a multiple of 4
// Each Cuda block handles 4 elements per thread to increase work per thread
//   and uses wider accesses to memory.
// Each Cuda block has 64 threads in y dimension (rows)
// and 16 threads in x dimension (columns)
// Each thread generates elements C[i][j] ... C[i][j+3] of the product
#define NROW 64
#define NCOL 16
// Structure data as float4's, with NCOL of them in each column
union mdata_t {
    float f[4];
    float4 f4;
};
__global__ void
cudaBlockQuadKernel(int N, float* dmatA, float* dmatB, float * dmatC) {
    // Prefix Key:
    // s: scaled.  Divided by 4.  Used when indexing columns
    // b: block.   Used to refer to elements within block
    // No prefix.  Used to refer to elements in global array
    //
    // Indexes into row of array
    int i = blockIdx.y * blockDim.y + threadIdx.y;  
    // Indexes into column, but in units of float4's
    int sj = blockIdx.x * blockDim.x + threadIdx.x; 
    int bi = threadIdx.y;   // Ranges between 0 and NROW-1
    int sbj = threadIdx.x;  // Ranges between 0 and NCOL-1
    int sN = N/4;           // Number of float4's in each row of matrices
    
    // Representing source & destination matrices as float4's:
    float4 *matAf4 = (float4 *) dmatA;
    float4 *matBf4 = (float4 *) dmatB;
    float4 *matCf4 = (float4 *) dmatC;

    /* Accumulate 4 elements in row of C */
    mdata_t sums;
    sums.f[0] = sums.f[1] = sums.f[2] = sums.f[3] = 0.0;
    mdata_t zeros;
    zeros.f[0] = zeros.f[1] = zeros.f[2] = zeros.f[3] = 0.0;
    
    // Shared space for two submatrices of A and B
    __shared__ mdata_t subA[NROW*NCOL];
    __shared__ mdata_t subB[NROW*NCOL];


    // Loop over k to compute product of all submatrices A[i][k] and B[k][j]
    for (int sk = 0; sk < sN; sk += NCOL) {
	int k = sk * 4;
	// Read the two submatrices from global memory
	if (i < N && sk+sbj < sN)
	    subA[RM(bi,sbj,NCOL)].f4 = matAf4[RM(i,sk+sbj,sN)];
	else
	    subA[RM(bi,sbj,NCOL)].f4 = zeros.f4;

	if (sj < sN && k+bi < N)
	    subB[RM(bi,sbj,NCOL)].f4 = matBf4[RM(k+bi,sj,sN)];
	else
	    subB[RM(bi,sbj,NCOL)].f4 = zeros.f4;

	// Wait until entire block gets filled
	__syncthreads();

	// Generate contribution to C[i][4*sj] .. C[i][4*sj+3]
	for (int sbk = 0; sbk < NCOL; sbk++) {
	    int bk = 4*sbk;
	    mdata_t a = subA[RM(bi,sbk,NCOL)];
	    mdata_t bfill[4];
	    bfill[0] = subB[RM(bk+0,sbj,NCOL)];
	    bfill[1] = subB[RM(bk+1,sbj,NCOL)];
	    bfill[2] = subB[RM(bk+2,sbj,NCOL)];
	    bfill[3] = subB[RM(bk+3,sbj,NCOL)];
	    float *b = (float *) &bfill;
	    for (int tj = 0; tj < 4; tj++) {
		sums.f[tj] +=
		    a.f[0] * b[RM(0,tj,4)]  +
		    a.f[1] * b[RM(1,tj,4)] +
		    a.f[2] * b[RM(2,tj,4)]  +
		    a.f[3] * b[RM(3,tj,4)];
	    }
	}
	// Wait until all products computed
	__syncthreads();
    }
    /* Store 4 elements into C */
    if (i < N && sj < sN)
	matCf4[RM(i,sj,sN)] = sums.f4;
}

// nVidia kernel.  Only works when N multiple of block size
#define BLOCK_SIZE LBLK

__global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}


__global__ void
cudaSmallBlockKernel(int N, float *dmatA, float *dmatB, float *dmatC) {
    // Assume that thread block contains submatrix of size SBLK x SBLK
    // Have SBLK extra threads available to serve as third index.
    // These are all within a single warp
    int bk = threadIdx.x;  // Range within single warp
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int bi = threadIdx.z;
    int bj = threadIdx.y;

    // Shared space for two submatrices of A and B
    __shared__ float subA[SBLK*SBLK];
    __shared__ float subB[SBLK*SBLK];
    // Shared space for partial products
    __shared__ float vals[SBLK*SBLK*SBLK];

    float sum = 0.0;

    // Loop over k to compute product of all submatrices A[i][k] and B[k][j]
    for (int k = 0; k < N; k+= SBLK) {
	// Designate threads with bk == 0 to fill A
	if (bk == 0) {
	    // Grab the two submatrices
	    if (i < N && k+bj < N)
		subA[RM(bi,bj,SBLK)] = dmatA[RM(i,k+bj,N)];
	    else
		subA[RM(bi,bj,SBLK)] = 0.0;

	}
	// Designate threads with bk == 1 to fill B
	if (bk == 1) {
	    if (j < N && k+bi < N)
		subB[RM(bi,bj,SBLK)] = dmatB[RM(k+bi,j,N)];
	    else
		subB[RM(bi,bj,SBLK)] = 0.0;

	}
	// Wait until entire block gets filled
	__syncthreads();

	// Compute all partial products of the submatrices
	vals[RM3(bi,bj,bk,SBLK)] = subA[RM(bi,bk,SBLK)] * subB[RM(bk,bj,SBLK)];
	// Wait until partial products computed
	__syncthreads();

	// Sum the values across the value of bk using tree reduction
	// These are all in same warp, and so don't require synchronization
	if (bk % 2  == 0)
	    vals[RM3(bi,bj,bk,SBLK)] += vals[RM3(bi,bj,bk+1,SBLK)];
	if (bk % 4 == 0)
	    vals[RM3(bi,bj,bk,SBLK)] += vals[RM3(bi,bj,bk+2,SBLK)];
	if (bk % 8 == 0) {
	    vals[RM3(bi,bj,bk,SBLK)] += vals[RM3(bi,bj,bk+4,SBLK)];
	    sum += vals[RM3(bi,bj,bk,SBLK)];
	}
	__syncthreads();
    }
    if (i < N && j < N && bk == 0)
	dmatC[RM(i,j,N)] = sum;
}


/* Preallocated blocks */
static int allocN = -1;
static float *aDevData = NULL;
static float *bDevData = NULL;
static float *tDevData = NULL;
static float *gDevData = NULL;
static float *sDevData = NULL;
static float *tHostData = NULL;
static float *gHostData = NULL;

void cudaSetup(int N, float *aData, float *bData, float *gData) {
    if (allocN == N)
	return;
    if (allocN > 0) {
	cudaFree(sDevData);
	cudaFree(aDevData);
	cudaFree(bDevData);
	cudaFree(tDevData);
	cudaFree(gDevData);
    }
    if (N > 0) {
	cudaMalloc((void **) &aDevData, N*N * sizeof(float));
	cudaMalloc((void **) &bDevData, N*N * sizeof(float));
	cudaMalloc((void **) &tDevData, N*N * sizeof(float));
	cudaMalloc((void **) &sDevData, N*N * sizeof(float));
	tHostData = (float *) calloc(N*N, sizeof(float));
    }
    gHostData = gData;
    cudaMemcpy(aDevData, aData, N*N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bDevData, bData, N*N * sizeof(float), cudaMemcpyHostToDevice);
    allocN = N;
}

// Get scratch for matrix
static float *cudaScratchMatrix(int N) {
    if (allocN != N) {
	setup(N);
    }
    return sDevData;
}

void cudaMultMatrixSimpleOld(int N, float *dmatA, float *dmatB, float *dmatC)
{
    dim3 threadsPerBlock(LBLK, LBLK);
    dim3 blocks(updiv(N, LBLK), updiv(N, LBLK));
    cudaSimpleKernelOld<<<blocks, threadsPerBlock>>>(N, dmatA, dmatB, dmatC);
}

void cudaMultMatrixSimple(int N, float *dmatA, float *dmatB, float *dmatC)
{
    dim3 threadsPerBlock(LBLK, LBLK);
    dim3 blocks(updiv(N, LBLK), updiv(N, LBLK));
    cudaSimpleKernel<<<blocks, threadsPerBlock>>>(N, dmatA, dmatB, dmatC);
}

void cudaMultMatrixTransposed(int N, float *dmatA, float *dmatB,
			      float *dmatC)
{
    dim3 threadsPerBlock(LBLK, LBLK);
    dim3 blocks(updiv(N, LBLK), updiv(N, LBLK));
    float *tranB = cudaScratchMatrix(N);
    cudaTransposeKernel<<<blocks, threadsPerBlock>>>(N, dmatB, tranB);
    cudaTransposedKernel<<<blocks, threadsPerBlock>>>(N, dmatA,
						      tranB, dmatC);
}

void cudaMultMatrixBlocked(int N, float *dmatA, float *dmatB, float *dmatC)
{
    dim3 threadsPerBlock(LBLK, LBLK);
    dim3 blocks(updiv(N, LBLK), updiv(N, LBLK));
    cudaBlockKernel<<<blocks, threadsPerBlock>>>(N, dmatA, dmatB, dmatC);
}

void cudaMultMatrixBlockedOld(int N, float *dmatA, float *dmatB,
			      float *dmatC)
{
    dim3 threadsPerBlock(LBLK, LBLK);
    dim3 blocks(updiv(N, LBLK), updiv(N, LBLK));
    cudaBlockKernelOld<<<blocks, threadsPerBlock>>>(N, dmatA, dmatB, dmatC);
}

void cudaMultMatrixBlockedQuad(int N, float *dmatA, float *dmatB,
			       float *dmatC)
{
    dim3 threadsPerBlock(NCOL,NROW);
    // Have same N/NROW blocks in both dimensions,
    // since each block computes NROW x NROW portion of product
    dim3 blocks(updiv(N, NROW), updiv(N, NROW));
    cudaBlockQuadKernel<<<blocks, threadsPerBlock>>>(N, dmatA, dmatB, dmatC);
}


void cudaMultMatrixNvidia(int N, float *dmatA, float *dmatB, float *dmatC)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(updiv(N, BLOCK_SIZE), updiv(N, BLOCK_SIZE));
    matrixMulCUDA<<<blocks, threadsPerBlock>>>(dmatC, dmatA, dmatB, N, N);
}

void cudaMultMatrixSmallBlocked(int N, float *dmatA, float *dmatB,
				float *dmatC)
{
    dim3 threadsPerBlock(SBLK, SBLK, SBLK);
    dim3 blocks(1, updiv(N, SBLK), updiv(N, SBLK));
    cudaSmallBlockKernel<<<blocks, threadsPerBlock>>>(N, dmatA,
						      dmatB, dmatC);
}


static int cudaRunMM(int N, mmul_t method) {
    switch (method) {
    case MMUL_CUDA_OLD_REFERENCE:
	cudaMultMatrixSimpleOld(N, aDevData, bDevData, tDevData);
	break;
    case MMUL_CUDA_REFERENCE:
	cudaMultMatrixSimple(N, aDevData, bDevData, tDevData);
	break;
    case MMUL_CUDA_TRANSPOSE:
	cudaMultMatrixTransposed(N, aDevData, bDevData, tDevData);
	break;
    case MMUL_CUDA_BLK:
	cudaMultMatrixBlocked(N, aDevData, bDevData, tDevData);
	break;
    case MMUL_CUDA_OLD_BLK:
	cudaMultMatrixBlockedOld(N, aDevData, bDevData, tDevData);
	break;
    case MMUL_CUDA_NVIDIA:
	cudaMultMatrixNvidia(N, aDevData, bDevData, tDevData);
	break;
    case MMUL_CUDA_SMALL_BLK:
	cudaMultMatrixSmallBlocked(N, aDevData, bDevData, tDevData);
	break;
    case MMUL_CUDA_QUAD_BLK:
	cudaMultMatrixBlockedQuad(N, aDevData, bDevData, tDevData);
	break;
    default:
	fprintf(stderr, "Haven't implemented method yet\n");
	return 0;
    }
    return 1;
}

double cudaBenchMM(int N, mmul_t method) {
    // Should already have done the setup
    if (allocN != N) {
	setup(N);
    }
    if (!cudaRunMM(N, method))
	return 1000.0;
    cudaMemcpy(tHostData, tDevData, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    if (checkMatrix(N, tHostData, gHostData) > 0)
	return 1000.0;
    /* Now do the real benchmarking */
    long ops = (long) 2 * N * N * N;
    long runs = (targetOps+ops-1)/ops;
    double startTime = CycleTimer::currentSeconds();
    for (long r = 0; r < runs; r++)
	cudaRunMM(N, method);
    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double ms = (endTime - startTime) * 1000.0;
    double gflops = (long) (runs*ops)/ms * 1e-6;
    fprintf(stderr, "%ld runs, %ld ops/run, %.2f ms, %.3f GFlops\n",
            runs, ops, ms, gflops);
    return gflops;
}


void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
