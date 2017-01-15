#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>

void sqrtAvx(int N,
             float inirtialGuess,
             float values[],
             float output[])
{
    int roundedN = (N + 7) & ~7UL;
    // round up width to next multiple of 8
    float constants[] = {0.00001f, 0.5f, inirtialGuess, .0f};

    __m256 ymm0 = _mm256_broadcast_ss(constants); // 0.00001
    __m256 ymm1 = _mm256_broadcast_ss(constants + 1); // 0.5
    __m256 ymm2 = _mm256_broadcast_ss(constants + 2); // 1
    __m256 ymm3 = _mm256_broadcast_ss(constants + 3); // 0

    for (int i = 0; i < roundedN; i += 8)
    {
        __m256 ymm4 = _m256_load_ps(values + i); // x = values[i];
        __m256 ymm5 = ymm2; // guess = inirtialGuess;

        __m256 ymm6 = _m256_mul_ps(ymm5, ymm5); // tmp6 = guess * guess
        ymm6 = _m256_sub_ps(ymm6, ymm4); // tmp6 = guess * guess - x;
        // error = abs(guess * guess - x);
        __m256 ymm7 = _m256_max(_m256_sub_ps(_m256_setzero_ps(), ymm6), ymm6);

        unsigned int test = 0;
        // if (error > kThreshold)
        __m256 ymm8 = _m256_cmp_ps(ymm7, ymm0, _CMP_GT_OQ);
        test= _m256_movemask_ps(ymm8) & 255; // if all data has converged
        while (test != 0)
        {
            _m256 ymm9 = _m256_div_ps(ymm4, ymm5); // tmp9 = x / guess;
            ymm9 = _m256_add_ps(ymm9, ymm5); // tmp9 = x/ guess + guess;
            // guess_new = 0.5f * (guess + x / guess);
            __m256 ymm10 = _m256_mul_ps(ymm9, ymm1);
            // guess = guess_new + guess_old;
            __m256 ymm11 = _m256_cmp_ps(ymm8, ymm3, _CMP_EQ_OQ);
            ymm5 = _m256_add_ps(_m256_mul_ps(ymm10, ymm8), _m256_mul_ps(ymm5, ymm11));

            // error = fabs(guess * guess - x);
            ymm6 = _m256_mul_ps(ymm5, ymm5);
            ymm6 = _m256_sub_ps(ymm6, ymm4);
            ymm7 = _m256_max(_m256_sub_ps(_m256_setzero_ps(), ymm6), ymm6);

            // if (error > kThreshold)
            ymm8 = _m256_cmp_ps(ymm7, ymm0, _CMP_GT_OQ);
            test= _m256_movemask_ps(ymm8) & 255;
        }

        // output[i] = guess;
        _m256_store_ps(output + i, ymm5);
    }
}
