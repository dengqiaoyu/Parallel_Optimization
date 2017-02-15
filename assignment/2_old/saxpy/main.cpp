#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

#include "CycleTimer.h"

void saxpyCuda(int N, float alpha, float* x, float* y, float* result);
void saxpySerial(int N, float scale, float X[], float Y[], float result[]);
void printCudaInfo();


// return GB/s
float toBW(int bytes, float sec) {
    return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

static float
toGFLOPS(int ops, float sec) {
    return static_cast<float>(ops) / 1e9 / sec;
}

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -n  --arraysize <INT>  Number of elements in arrays\n");
    printf("  -?  --help             This message\n");
}


int main(int argc, char** argv) {

    int N = 20 * 1000 * 1000;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"arraysize",  1, 0, 'n'},
        {"help",       0, 0, '?'},
        {0 , 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "?n:", long_options, NULL)) != EOF) {

        switch (opt) {
        case 'n':
            N = atoi(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////

    unsigned int TOTAL_BYTES = 4 * N * sizeof(float);
    unsigned int TOTAL_FLOPS = 2 * N;

    const float alpha = 2.0f;
    float* xarray = new float[N];
    float* yarray = new float[N];
    float* resultarray = new float[N];

    // load X, Y, store result
    for (int i = 0; i < N; i++) {
        xarray[i] = yarray[i] = i % 10;
        resultarray[i] = 0.f;
    }

    printCudaInfo();

    printf("CUDA saxpy:\n");
    for (int i = 0; i < 3; i++) {
        printf("\nRound %d\n", i);
        saxpyCuda(N, alpha, xarray, yarray, resultarray);
    }

    for (int i = 0; i < N; i++) {
        resultarray[i] = 0.f;
    }

    printf("\nSerial Saxpy:\n");
    double serialDuration = 0.f;
    for (int i = 0; i < 3; i++) {
        printf("\nRound %d\n", i);
        double startTime = CycleTimer::currentSeconds();
        saxpySerial(N, alpha, xarray, yarray, resultarray);
        double endTime = CycleTimer::currentSeconds();
        serialDuration = endTime - startTime;
        printf("[saxpy serial]:\t\t[%.3f] ms\t[%.3f] GB/s\t[%.3f] GFLOPS\n",
               serialDuration * 1000,
               toBW(TOTAL_BYTES, serialDuration),
               toGFLOPS(TOTAL_FLOPS, serialDuration));
    }

    delete [] xarray;
    delete [] yarray;
    delete [] resultarray;

    return 0;
}

void saxpySerial(int N, float scale, float X[], float Y[], float result[]) {
    for (int i = 0; i < N; i++) {
        result[i] = scale * X[i] + Y[i];
    }
}
