// CUDA Code Samples
// Carnegie Mellon University.  15-417, Spring 2017
// R. E. Bryant

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "CycleTimer.h"

// Support for CUDA error checking
// Wrapper for CUDA functions
#define CHK(ans) gpuAssert((ans), __FILE__, __LINE__);

// Check CUDA error code
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %s\n",
                cudaGetErrorString(code), file, line);
    }
}

// Cannot wrap kernel launches.  Instead, insert this after each
//   kernel launch.
#define POSTKERNEL CHK(cudaPeekAtLastError())

#define REDUCE 1
#include "reduce.h"

// Tunable parameters
int reduceDegree = 2;
float reduceTolerance = 0.001;
#if DEBUG
int reduceThreadsPerBlock = 32;
int reduceRuns = 1;
int reduceLength = 10;
#else
int reduceThreadsPerBlock = 1024;
int reduceRuns = 250;
int reduceLength = 1024 * 1024;
#endif


// Reducers
float inplaceReduce(int length, float *srcVecDevice,
                    float *scratchVecDevice);

// Set this to one of the above-listed reducers
#define REDUCER inplaceReduce

// Helper functions
static inline float toGF(int ops, float secs) {
    return ops / secs / (float) (1 << 30);
};

// Integer division, rounding up
#define UPDIV(n, d) (((n)+(d)-1)/(d))

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

// Reducer based on inplace reduction
float
inplaceReduce(int length, float *srcVecDevice, float *scratchVecDevice) {
    int threadsPerBlock = reduceThreadsPerBlock;
    int degree = reduceDegree;
    float val;
    int nlength;
    // Copy source data into scratch area so that can modify it
    CHK(cudaMemcpy(scratchVecDevice, srcVecDevice,
                   length * sizeof(float),
                   cudaMemcpyDeviceToDevice));
    for (; length > 1; length = nlength) {
        nlength = UPDIV(length, degree);
        int blocks = UPDIV(nlength, threadsPerBlock);
        inplaceReduceKernel <<< blocks, threadsPerBlock>>>(length, nlength,
                scratchVecDevice);
        POSTKERNEL;
        CHK(cudaDeviceSynchronize());
    }
    CHK(cudaMemcpy(&val, &scratchVecDevice[0],
                   sizeof(float), cudaMemcpyDeviceToHost));
    return val;
}
/* insert_break */
double cudaReduce(int length, float *aVec, float targetVal) {
    float *aVecDevice;
    float *scratchVecDevice;
    CHK(cudaMalloc((void **)&aVecDevice, length * sizeof(float)));
    CHK(cudaMemcpy(aVecDevice, aVec, length * sizeof(float),
                   cudaMemcpyHostToDevice));
    CHK(cudaMalloc((void **)&scratchVecDevice,
                   (length + 32) * sizeof(float)));

    float startTime = CycleTimer::currentSeconds();
    for (int r = 0; r < reduceRuns; r++) {
        float val = REDUCER(length, aVecDevice, scratchVecDevice);
        checkResult(val, targetVal);
    }
    float endTime = CycleTimer::currentSeconds();
    float gflops = toGF(length * reduceRuns, endTime - startTime);
    CHK(cudaFree(aVecDevice));
    CHK(cudaFree(scratchVecDevice));
    return gflops;
}
