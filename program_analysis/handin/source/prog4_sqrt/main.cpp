#include <stdio.h>
#include <algorithm>
#include <pthread.h>
#include <math.h>
#include <errno.h>

#include "CycleTimer.h"
#include "sqrt_ispc.h"

using namespace ispc;

extern void sqrtSerial(int N, float startGuess, float* values, float* output);
extern void sqrtSerialV2(int N, float startGuess, float* values, float* output);
extern void sqrtAVX(int N, float startGuess, float* values, float* output);


static void verifyResult(int N, float* result, float* gold) {
    for (int i=0; i<N; i++) {
        if (fabs(result[i] - gold[i]) > 1e-4) {
            printf("Error: [%d] Got %f expected %f\n", i, result[i], gold[i]);
        }
    }
}

int main() {

    const unsigned int N = 20 * 1000 * 1000;
    const float initialGuess = 1.0f;

    float* values = NULL;
    char ret = 0;
    ret = posix_memalign((void **)&values, 32, sizeof(float) * N);
    if (ret != 0)
    {
        printf("%s\n", strerror(ret));
        return 1;
    }

    float* output = NULL;
    ret = posix_memalign((void **)&output, 32, sizeof(float) * N);
    if (ret != 0)
    {
        printf("%s\n", strerror(ret));
        return 1;
    }

    float* gold = new float[N];

    for (unsigned int i=0; i<N; i++)
    {
        // random input values
        values[i] = .001f + 2.998f * static_cast<float>(rand()) / RAND_MAX;

        // ##### For best case in ISPC #####
        // values[i] = 2.998f;
        // ##### For best case in ISPC #####

        // ##### For worst case in ISPC #####
        //if (i % 4 == 0)
        //{
        //    values[i] = 2.998f;
        //}
        //else
        //{
        //    values[i] = 1.0f;
        //}
        // ##### For worst case in ISPC #####
        output[i] = 0.f;
    }

    // generate a gold version to check results
    for (unsigned int i=0; i<N; i++)
        gold[i] = sqrt(values[i]);

    //
    // And run the serial implementation 3 times, again reporting the
    // minimum time.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrtSerial(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[sqrt serial]:\t\t[%.3f] ms\n", minSerial * 1000);

    verifyResult(N, output, gold);

    for (unsigned int i = 0; i < N; ++i)
        output[i] = 0;


    // sqrtSerialV2
    double minSerialV2 = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrtSerialV2(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minSerialV2 = std::min(minSerialV2, endTime - startTime);
    }

    printf("[sqrt serialV2]:\t[%.3f] ms\n", minSerialV2 * 1000);

    verifyResult(N, output, gold);

    printf("\t\t\t\t(%.2fx speedup from serialV2)\n", minSerial/minSerialV2);

    for (unsigned int i = 0; i < N; ++i)
        output[i] = 0;




    //
    // Compute the image using the ispc implementation; report the minimum
    // time of three runs.
    //
    double minISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minISPC = std::min(minISPC, endTime - startTime);
    }

    printf("[sqrt ispc]:\t\t[%.3f] ms\n", minISPC * 1000);

    verifyResult(N, output, gold);

    // Clear out the buffer
    for (unsigned int i = 0; i < N; ++i)
        output[i] = 0;

    //
    // Tasking version of the ISPC code
    //
    double minTaskISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc_withtasks(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minTaskISPC = std::min(minTaskISPC, endTime - startTime);
    }

    printf("[sqrt task ispc]:\t[%.3f] ms\n", minTaskISPC * 1000);

    verifyResult(N, output, gold);

    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minSerial/minISPC);
    printf("\t\t\t\t(%.2fx speedup from task ISPC)\n", minSerial/minTaskISPC);

    // Clear out the buffer
    for (unsigned int i = 0; i < N; ++i)
        output[i] = 0;

    //
    // Tasking version of the AVX code
    //
    double minAVX = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrtAVX(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minAVX = std::min(minAVX, endTime - startTime);
    }

    printf("[sqrt avx]:\t\t[%.3f] ms\n", minAVX * 1000);

    verifyResult(N, output, gold);

    printf("\t\t\t\t(%.2fx speedup from AVX over Serial)\n", minSerial/minAVX);
    printf("\t\t\t\t(%.2fx speedup from AVX over ISPC)\n", minISPC/minAVX);
    printf("\t\t\t\t(%.2fx speedup from AVX over ISPC task)\n", minTaskISPC/minAVX);
    printf("\t\t\t\t(%.2fx speedup from AVX over SerialV2)\n", minSerialV2/minAVX);


    free(values);
    free(output);
    delete[] gold;

    return 0;
}
