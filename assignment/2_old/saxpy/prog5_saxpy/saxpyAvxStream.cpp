/**
 * @name: saxpyAvxStream
 * @brief: Implement saxpy with AVX instructions with stream instructions which
 *         can using a non-temporal memory hint to reduce memory referencing
 * @author: Qiaoyu Deng(qdeng@andrew.cmu)
 */
#include <immintrin.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct {
    int startIndex, endIndex;
    float scale;
    float* X;
    float* Y;
    float* result;
} WorkerArgs;

void *workThreadStart(void* threadArgs)
{
    WorkerArgs* args = static_cast<WorkerArgs*>(threadArgs);
    static const int VECTOR_WIDTH = 8;

    int startIndex = args->startIndex;
    int endIndex = args->endIndex;
    float scale = args->scale;
    float* X = args->X;
    float* Y = args->Y;
    float* result = args->result;

    // Change from a int to a float vector
    __m256 scaleVector = _mm256_set1_ps(scale);

    for (int i = startIndex; i < endIndex; i += VECTOR_WIDTH)
    {
        __m256 XVector = _mm256_load_ps(X + i); // X[i]
        __m256 YVector = _mm256_load_ps(Y + i); // Y[i]
        // resultTmp = scale * X[i];
        __m256 resultVector = _mm256_mul_ps(scaleVector, XVector);
        // resultTmp = scale * X[i] + Y[i];
        resultVector = _mm256_add_ps(resultVector, YVector);
        // result[i] = resultTmp;
        _mm256_stream_ps(result + i, resultVector);
    }

    return NULL;
}

void saxpyAvxStream(int N,
                    float scale,
                    float X[],
                    float Y[],
                    float result[])
{
    const int numThreads = 6;
    const static int MAX_THREADS = 32;
    // Check whether array can be divided by the number of threads
    // without reminder.
    int residue = N % numThreads;
    int blockSize = N / numThreads;
    // Since AVX we will use will compute 8 floats at the same time, we need
    // to ensure that every block can be divided by 8 exactly.
    int roundedBlockSize= (blockSize + 7) & ~7UL;

    pthread_t workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i = 0; i < numThreads; i++)
    {
        // The last thread might be assigned the remaining part of array, which
        // might not be divided by 8 without reminder.
        if (i == numThreads - 1 && residue != 0)
        {
            args[i].startIndex = i * roundedBlockSize;
            args[i].endIndex = N;
        }
        else
        {
            // Assign every block to every thread.
            args[i].startIndex = i * roundedBlockSize;
            args[i].endIndex = (i + 1) * roundedBlockSize;
        }
        args[i].scale = scale;
        args[i].X = X;
        args[i].Y = Y;
        args[i].result = result;
    }

    for (int i = 1; i < numThreads; i++)
    {
        pthread_create(&workers[i], NULL, workThreadStart, &args[i]);
    }

    workThreadStart(&args[0]);

    for (int i = 1; i < numThreads; i++)
    {
        pthread_join(workers[i], NULL);
    }
}
