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
    static int VECTOR_WIDTH = 8;
    int startIndex = args->startIndex;
    int endIndex = args->endIndex;
    float scale = args->scale;
    float* X = args->X;
    float* Y = args->Y;
    float* result = args->result;

    __m256 scaleVector = _mm256_set1_ps(scale);

    printf("line 27\n");
    for (int i = startIndex; i < endIndex; i += VECTOR_WIDTH)
    {
        __m256 XVector = _mm256_load_ps(X + i);
        __m256 YVector = _mm256_load_ps(Y + i);
        __m256 resultVector = _mm256_fmadd_ps(scaleVector, XVector, YVector);

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
    const int numThreads = 2;
    const static int MAX_THREADS = 32;
    int residue = N % numThreads;
    int blockSize = N / numThreads;
    int roundedBlockSize= (blockSize + 7) & ~7UL;
    printf("roundedBlockSize: %d\n", roundedBlockSize);
    pthread_t workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    printf("line 54\n");
    for (int i = 0; i < numThreads; i++)
    {
        if (i == numThreads - 1 && residue != 0)
        {
            args[i].startIndex = i * roundedBlockSize;
            args[i].endIndex = N;
        }
        else
        {
            args[i].startIndex = i * roundedBlockSize;
            args[i].endIndex = (i + 1) * roundedBlockSize;
        }
        args[i].scale = scale;
        args[i].X = X;
        args[i].Y = Y;
        args[i].result = result;
    }
    printf("line 72\n");

    for (int i = 1; i < numThreads; i++)
    {
        pthread_create(&workers[i], NULL, workThreadStart, &args[i]);
    }

    workThreadStart(&args[0]);
    printf("line 78");

    for (int i = 1; i < numThreads; i++)
    {
        pthread_join(workers[i], NULL);
    }
}
