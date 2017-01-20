/**
 * @name: sqrtAVX
 * @brief: Implement sqrt function by newton method, and improve performance by
 *         changing some calculating step
 * @author: Qiaoyu Deng(qdeng@andrew.cmu.edu)
 */
#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>

void sqrtAVX(int N,
             float inirtialGuess,
             float values[],
             float output[])
{
    int roundedN = (N + 7) & ~7UL;
    // round up width to next multiple of 8
    float constants[] = {0.00001f, 0.5f, inirtialGuess, .0f};

    __m256 epsilon = _mm256_broadcast_ss(constants); // 0.00001
    __m256 oneHalf = _mm256_broadcast_ss(constants + 1); // 0.5
    __m256 iniGuess = _mm256_broadcast_ss(constants + 2); // 1
    __m256 zero = _mm256_broadcast_ss(constants + 3); // 0

    for (int i = 0; i < roundedN; i += 8)
    {
        __m256 xVec = _mm256_load_ps(values + i); // x = values[i];
        __m256 guess = iniGuess; // guess = inirtialGuess;

        __m256 tmp = _mm256_mul_ps(guess, guess); // tmp6 = guess * guess
        tmp = _mm256_sub_ps(tmp, xVec); // tmp6 = guess * guess - x;
        // error = abs(guess * guess - x);
        __m256 error = _mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), tmp), tmp);

        unsigned int test = 0;
        // if (error > kThreshold)
        __m256 ifMask = _mm256_cmp_ps(error, epsilon, _CMP_GT_OQ);
        test= _mm256_movemask_ps(ifMask) & 255; // if all data has converged
        while (test != 0)
        {
            tmp = _mm256_div_ps(xVec, guess); // tmp9 = x / guess;
            tmp = _mm256_add_ps(tmp, guess); // tmp9 = x/ guess + guess;
            // guess_new = 0.5f * (guess + x / guess);
            __m256 guess_new = _mm256_mul_ps(tmp, oneHalf);
            // guess = guess_new + oldMask;
            __m256 oldMask = _mm256_cmp_ps(ifMask, zero, _CMP_EQ_OQ);
            guess = _mm256_add_ps(_mm256_and_ps(guess_new, ifMask), _mm256_and_ps(guess, oldMask));

            // error = fabs(guess * guess - x);
            tmp = _mm256_mul_ps(guess, guess);
            tmp = _mm256_sub_ps(tmp, xVec);
            error = _mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), tmp), tmp);

            // if (error > kThreshold)
            ifMask = _mm256_cmp_ps(error, epsilon, _CMP_GT_OQ);
            test= _mm256_movemask_ps(ifMask) & 255;
        }

        // output[i] = guess;
        _mm256_store_ps(output + i, guess);
    }
}
