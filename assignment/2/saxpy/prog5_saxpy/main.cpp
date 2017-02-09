#include <stdio.h>
#include <algorithm>
#include <errno.h>

#include "CycleTimer.h"
#include "saxpy_ispc.h"

extern void saxpySerial(int N, float a, float* X, float* Y, float* result);
extern void saxpyAvxStream(int N, float a, float* X, float* Y, float* result);


// return GB/s
static float
toBW(int bytes, float sec) {
    return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

static float
toGFLOPS(int ops, float sec) {
    return static_cast<float>(ops) / 1e9 / sec;
}

using namespace ispc;


int main() {

    const unsigned int N = 20 * 1000 * 1000; // 20 M element vectors (~80 MB)
    const unsigned int TOTAL_BYTES = 4 * N * sizeof(float);
    const unsigned int TOTAL_FLOPS = 2 * N;

    float scale = 2.f;

    // To use posix_memalign, memory be aligned on a 32-byte boundary
    float* arrayX = new float[N];
    char ret = 0;
    float* arrayXAlign = NULL;
    ret = posix_memalign((void **)&arrayXAlign, 32, sizeof(float) * N);
    if (ret != 0)
    {
        printf("%s\n", strerror(ret));
        return 1;
    }

    float* arrayY = new float[N];
    float* arrayYAlign = NULL;
    ret = posix_memalign((void **)&arrayYAlign, 32, sizeof(float) * N);
    if (ret != 0)
    {
        printf("%s\n", strerror(ret));
        return 1;
    }

    float* result = new float[N];
    float* resultAlign;
    ret = posix_memalign((void **)&resultAlign, 32, sizeof(float) * N);
    if (ret != 0)
    {
        printf("%s\n", strerror(ret));

        return 1;
    }

    // initialize array values
    for (unsigned int i=0; i<N; i++)
    {
        arrayX[i] = i;
        arrayXAlign[i] = i;
        arrayY[i] = i;
        arrayYAlign[i] = i;
        result[i] = 0.f;
        resultAlign[i] = 0.f;
    }

    //
    // Run the serial implementation. Repeat three times for robust
    // timing.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime =CycleTimer::currentSeconds();
        saxpySerial(N, scale, arrayX, arrayY, result);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[saxpy serial]:\t\t[%.3f] ms\t[%.3f] GB/s\t[%.3f] GFLOPS\n",
           minSerial * 1000,
           toBW(TOTAL_BYTES, minSerial),
           toGFLOPS(TOTAL_FLOPS, minSerial));

    // Clear out the buffer
    for (unsigned int i = 0; i < N; ++i)
        result[i] = 0.f;

    //
    // Run the ISPC (single core) implementation
    //
    double minISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        saxpy_ispc(N, scale, arrayX, arrayY, result);
        double endTime = CycleTimer::currentSeconds();
        minISPC = std::min(minISPC, endTime - startTime);
    }

    printf("[saxpy ispc]:\t\t[%.3f] ms\t[%.3f] GB/s\t[%.3f] GFLOPS\n",
           minISPC * 1000,
           toBW(TOTAL_BYTES, minISPC),
           toGFLOPS(TOTAL_FLOPS, minISPC));

    // Clear out the buffer
    for (unsigned int i = 0; i < N; ++i)
        result[i] = 0.f;

    //
    // Run the ISPC (multi-core) implementation
    //
    double minTaskISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        saxpy_ispc_withtasks(N, scale, arrayX, arrayY, result);
        double endTime = CycleTimer::currentSeconds();
        minTaskISPC = std::min(minTaskISPC, endTime - startTime);
    }

    printf("[saxpy task ispc]:\t[%.3f] ms\t[%.3f] GB/s\t[%.3f] GFLOPS\n",
           minTaskISPC * 1000,
           toBW(TOTAL_BYTES, minTaskISPC),
           toGFLOPS(TOTAL_FLOPS, minTaskISPC));


    // Clear out the buffer
    for (unsigned int i = 0; i < N; ++i)
        result[i] = 0.f;

    //
    // Run the AVX implementation by using stream instruction
    //
    double minAvxStream = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        saxpyAvxStream(N, scale, arrayX, arrayY, resultAlign);
        double endTime = CycleTimer::currentSeconds();
        minAvxStream = std::min(minAvxStream, endTime - startTime);
    }

    printf("[saxpy with AVX stream]:\t[%.3f] ms\t[%.3f] GB/s\t[%.3f] GFLOPS\n",
           minAvxStream * 1000,
           toBW(TOTAL_BYTES, minAvxStream),
           toGFLOPS(TOTAL_FLOPS, minAvxStream));


    printf("\t\t\t\t(%.2fx speedup from AVX with stream)\n",
            minSerial/minAvxStream);
    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minSerial/minISPC);
    printf("\t\t\t\t(%.2fx speedup from task ISPC)\n", minSerial/minTaskISPC);
    printf("\t\t\t\t(%.2fx speedup from use of tasks over ISPC)\n",
            minISPC/minTaskISPC);


    delete[] arrayX;
    delete[] arrayY;
    delete[] result;
    free(arrayXAlign);
    free(arrayYAlign);
    free(resultAlign);

    return 0;
}
