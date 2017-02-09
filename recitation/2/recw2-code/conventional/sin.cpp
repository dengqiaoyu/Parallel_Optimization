#include <stdio.h>
#include "sin.h"
#define MAXTERMS 30

// Original version of sin function
void sinx_reference(int N, int terms, float * x, float *result) {
    for (int i = 0; i < N; i++) {
        float value = x[i];
        float numer = x[i] * x[i] * x[i];
        int denom = 6; // 3!
        int sign = -1;

        for (int j = 1; j <= terms; j++) {
            value += sign * numer / denom;
            numer *= x[i] * x[i];
            denom *= (2 * j + 2) * (2 * j + 3);
            sign *= -1;
        }

        result[i] = value;
    }
}

// Optimized versions
void sinx_better(int N, int terms, float * x, float *result) {
    // Make some simple fixes that you think might help
    for (int i = 0; i < N; i++) {
        float value = x[i];

        // Your code here
        float numer = value * value * value;
        int denom = 6;
        int sign = -1;
        for (int j = 1; j <= terms; j++) {
            value += sign * numer / denom;
            numer *= value * value;
            int tmp = 2 * j + 2;
            denom *= (tmp) * (tmp + 1);
            sign = -sign;
            // Your code here
        }

        result[i] = value;
    }
}

// Try precomputing the reciprocal factorials
void sinx_predenom(int N, int terms, float * x, float *result) {
    float rdenom[MAXTERMS];

    // Your code here

    int sign = -1;
    for (int i = 0; i < terms; i++) {
        int tmp = 2 * i + 2;
        rdenom[i] = sign * tmp * (tmp + 1);
        sign = -sign;
    }

    for (int i = 0; i < N; i++) {
        float value = x[i];
        float numer = value * value * value;
        float numer_denom = numer / rdenom[i];
        // Your code here

        for (int j = 1; j <= terms; j++) {
            value += sign * numer_denom;
            // Your code here
        }
        result[i] = value;
    }
}

// Try precomputing the signed reciprocal factorials
void sinx_predenoms(int N, int terms, float * x, float *result) {
    float rdenom[MAXTERMS];

    // Your code here

    for (int i = 0; i < N; i++) {
        float value = x[i];

        // Your code here

        for (int j = 1; j <= terms; j++) {

            // Your code here

        }
        result[i] = value;
    }
}

// Try a simple unrolling by 2
void sinx_unrollx2(int N, int terms, float * x, float *result) {
    float rdenom[MAXTERMS];

    // Your code here

    for (int i = 0; i < N; i++) {
        float value = x[i];

        // Your code here

        int j;
        for (j = 1; j <= terms - 1; j += 2) {

            // Your code here

        }

        for (; j <= terms; j++) {

            // Your code here

        }

        result[i] = value;
    }
}

// Try unrolling x2, with reassociation and refactoring
void sinx_unrollx2a(int N, int terms, float * x, float *result) {
    float rdenom[MAXTERMS];

    // Your code here

    for (int i = 0; i < N; i++) {
        float value = x[i];

        // Your code here


        int j;
        for (j = 1; j <= terms - 1; j += 2) {

            // Your code here

        }

        for (; j <= terms; j++) {

            // Your code here

        }

        result[i] = value;
    }
}



void init_benchmarks(benchmark_t *b) {
    int i = 0;
    b[i].name = "reference"; b[i++].fun = sinx_reference;
    b[i].name = "better";    b[i++].fun = sinx_better;
    b[i].name = "predenom";    b[i++].fun = sinx_predenom;
    b[i].name = "predenoms";    b[i++].fun = sinx_predenoms;
    b[i].name = "unrollx2";    b[i++].fun = sinx_unrollx2;
    b[i].name = "unrollx2a";    b[i++].fun = sinx_unrollx2a;
    b[i].name = "";          b[i++].fun = NULL;
}

