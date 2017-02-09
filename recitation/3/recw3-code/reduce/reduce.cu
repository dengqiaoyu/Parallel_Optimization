// CUDA Code Samples
// Carnegie Mellon University.  15-417, Spring 2017
// R. E. Bryant

#include <cuda.h>
#include <cuda_runtime.h>
#include "CycleTimer.h"

#define REDUCE 1
#include "reduce.h"

// Tunable parameters
int reduceDegree = 2;
int reduceThreadsPerBlock = 1024;
int reduceRuns = 250;
float reduceTolerance = 0.001;

// Reducers
float inplaceReduce(int length, float *srcVecDevice,
		    float *scratchVecDevice);
float copyReduce(int length, float *srcVecDevice,
		 float *scratchVecDevice);
float blockReduce(int length, float *srcVecDevice,
		  float *scratchVecDevice);
float blockCollectReduce(int length, float *srcVecDevice,
			 float *scratchVecDevice);

// Set this to one of the above-listed reducers
#define REDUCER inplaceReduce

// Helper functions
static inline float toGF(int ops, float secs) {
    return ops/secs/(float) (1 << 30);
};

// Integer division, rounding up
#define UPDIV(n, d) (((n)+(d)-1)/(d))

// Minimum/Maximum
#define MIN(x, y) ((x)<(y)?(x):(y))
#define MAX(x, y) ((x)>(y)?(x):(y))

/* insert_break */
////////////////////////////////////////////////////////////////////////
// Reducer Example #1.  Inplace reduction

// CUDA kernel
// Destructively reduce data from length to nlength.
__global__ void
inplaceReduceKernel(int length, int nlength, float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nlength) {
	float val = data[idx];
	for (int i = idx + nlength; i < length; i += nlength)
	    val += data[i];
	data[idx] = val;
    }
}
/* insert_break */

// Reducer based on inplace reduction 
float
inplaceReduce(int length, float *srcVecDevice, float *scratchVecDevice) {
    int threadsPerBlock = reduceThreadsPerBlock;
    int degree = reduceDegree;
    float val;
    int nlength;
    // Copy source data into scratch area so that can modify it
    cudaMemcpy(scratchVecDevice, srcVecDevice, length * sizeof(float),
	       cudaMemcpyDeviceToDevice);
    for (; length > 1; length = nlength) {
	nlength = UPDIV(length, degree);
	int blocks = UPDIV(nlength, threadsPerBlock);
	inplaceReduceKernel<<<blocks, threadsPerBlock>>>(length, nlength,
						       scratchVecDevice);
	cudaThreadSynchronize();
    }
    cudaMemcpy(&val, &scratchVecDevice[0],
	       sizeof(float), cudaMemcpyDeviceToHost);
    return val;
}

/* insert_break */
////////////////////////////////////////////////////////////////////////
// Reducer Example #2.  Copying reduction

// CUDA kernel
// Reduce src from length to nlength without altering source
__global__ void
copyReduceKernel(int length, int nlength, float *src, float *dest) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nlength) {
    // Accumulate values for dest[idx]

    // Your code here

    }
}
/* insert_break */

// Reducer based on copying reductions
float
copyReduce(int length, float *srcVecDevice, float *scratchVecDevice) {
    int threadsPerBlock = reduceThreadsPerBlock;
    int degree = reduceDegree;
    float val;
    int sindex = 0;
    int nlength;
    float *src = srcVecDevice;
    float *dest;
    for (; length > 1; length = nlength) {
	nlength = UPDIV(length, degree);
	dest = &scratchVecDevice[sindex];
	int blocks = UPDIV(nlength, threadsPerBlock);
	copyReduceKernel<<<blocks, threadsPerBlock>>>(length, nlength,
						      src, dest);
	sindex += nlength;
	src = dest;
	cudaThreadSynchronize();
    }
    cudaMemcpy(&val, &dest[0], sizeof(float), cudaMemcpyDeviceToHost);
    return val;
}
/* insert_break */


/* insert_break */
////////////////////////////////////////////////////////////////////////
// Reducer Example #3.  Simple block-based reduction

// CUDA kernel
// Reduce portion of vector by threadsPerBlock
__global__ void
blockReduceKernel(int length, int degree, float *src, float *dest) {
    int idx = threadIdx.x;  // Thread ID within block
    int tidx = blockIdx.x;   // Identifies which tree this is part of
    int bsize = blockDim.x;
    int gidx = idx + blockDim.x * tidx;

    __shared__ float scratch[THREADSPERBLOCK];
    // First bring chunk of source array into shared memory
    float val = 0.0;

    if (gidx < length)
	val += src[gidx];

    scratch[idx] = val;
	    
    __syncthreads();

    int nlen;
    for (int len = bsize; len > 1; len = nlen) {
	nlen = UPDIV(len, degree);
	if (idx < nlen) {
	    float val = scratch[idx];
	    for (int i = idx + nlen; i < len; i += nlen)
		val += scratch[i];
	    scratch[idx] = val;
	}
	__syncthreads();
    }

    if (idx == 0)
	dest[tidx] = scratch[0];
}
/* insert_break */

float
blockReduce(int length, float *srcVecDevice, float *scratchVecDevice) {
    int threadsPerBlock = reduceThreadsPerBlock;
    int degree = reduceDegree;
    int bdegree = threadsPerBlock;
    int nlength;
    int sindex = 0;
    float *src = srcVecDevice;
    float *dest;
    for (; length > 1; length = nlength) {
	nlength = UPDIV(length, bdegree);
	dest = &scratchVecDevice[sindex];
	int blocks = UPDIV(length, bdegree);
	blockReduceKernel<<<blocks, threadsPerBlock>>>(length,
						  degree, src, dest);
	sindex += nlength;
	src = dest;
	cudaThreadSynchronize();
    }
    float val = 0.0;
    cudaMemcpy(&val, &dest[0], sizeof(float), cudaMemcpyDeviceToHost);
    return val;
}

/* insert_break */
////////////////////////////////////////////////////////////////////////
// Reducer Example #4.  Better block-based collection


// Reduce portion of vector by degree * threadsPerBlock
__global__ void
blockCollectReduceKernel(int length, int threads, int degree,
			 float *src, float *dest) {
    int idx = threadIdx.x;  // Thread ID within block
    int tidx = blockIdx.x;   // Identifies which tree this is part of
    int bsize = blockDim.x;
    int gidx = idx + blockDim.x * tidx;

    __shared__ float scratch[THREADSPERBLOCK];

    // First pass sums source values into shared memory
    float val = 0.0;

    // Accumulate values for scratch[idx]
    // by indexing src from gidx up to length
    // with a stride of threads

    // Your code here


    scratch[idx] = val;
    __syncthreads();

    int nlen;
    for (int len = bsize; len > 1; len = nlen) {
	nlen = UPDIV(len, degree);
	if (idx < nlen) {
	    float val = scratch[idx];
	    for (int i = idx + nlen; i < len; i += nlen)
		val += scratch[i];
	    scratch[idx] = val;
	}
	__syncthreads();
    }

    if (idx == 0)
	dest[tidx] = scratch[0];
}
/* insert_break */

float
blockCollectReduce(int length, float *srcVecDevice,
		   float *scratchVecDevice) {
    int threadsPerBlock = reduceThreadsPerBlock;
    int degree = reduceDegree;
    int bdegree = threadsPerBlock * degree;
    int nlength;
    int sindex = 0;
    float *src = srcVecDevice;
    float *dest;
    for (; length > 1; length = nlength) {
	nlength = UPDIV(length, bdegree);
	dest = &scratchVecDevice[sindex];
	int blocks = UPDIV(length, bdegree);
	int threads = blocks * threadsPerBlock;
	blockCollectReduceKernel<<<blocks, threadsPerBlock>>>(length,
					    threads, degree, src, dest);
	sindex += nlength;
	src = dest;
	cudaThreadSynchronize();
    }
    float val = 0.0;
    cudaMemcpy(&val, &dest[0], sizeof(float), cudaMemcpyDeviceToHost);
    return val;
}



double cudaReduce(int length, float *aVec, float targetVal) {
    float *aVecDevice;
    float *scratchVecDevice;
    cudaMalloc((void **)&aVecDevice, length * sizeof(float));
    cudaMemcpy(aVecDevice, aVec, length * sizeof(float),
	       cudaMemcpyHostToDevice);
    cudaMalloc((void **)&scratchVecDevice,
	       (length + 32) * sizeof(float));

    float startTime = CycleTimer::currentSeconds();
    for (int r = 0; r < reduceRuns; r++) {
	float val = REDUCER(length, aVecDevice, scratchVecDevice);
	checkResult(val, targetVal);
    }
    float endTime = CycleTimer::currentSeconds();
    float gflops = toGF(length * reduceRuns, endTime-startTime);
    cudaFree(aVecDevice);
    cudaFree(scratchVecDevice);
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
