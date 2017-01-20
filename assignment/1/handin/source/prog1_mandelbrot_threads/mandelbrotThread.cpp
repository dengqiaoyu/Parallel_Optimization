/**
 * @name: mandelbrotThread
 * @breif: Implement better load balance version of mandelbrotThread by assign
 *         work load column by column
 * @author: Qiaoyu Deng(qdeng@andrew.cmu.edu)
 */
#include <stdio.h>
#include <pthread.h>
#include <algorithm>

#include "CycleTimer.h"

typedef struct {
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int* output;
    int threadId;
    int numThreads;
} WorkerArgs;

static inline int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}


//
// workerThreadStart --
//
// Thread entrypoint.
void* workerThreadStart(void* threadArgs) {
    double minThread = 1e30;
    double startTime = CycleTimer::currentSeconds();

    WorkerArgs* args = static_cast<WorkerArgs*>(threadArgs);
    float x0 = args->x0;
    float y0 = args->y0;
    float x1 = args->x1;
    float y1 = args->y1;
    int width = args->width;
    int height = args->height;
    int maxIterations= args->maxIterations;
    int *output = args->output;
    int threadId = args->threadId;
    int numThreads = args->numThreads;

    int startRow = 0;
    int endRow = startRow + height;

    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;
    // Use Round Robin column by column
    for (int j = startRow; j < endRow; j++)
    {
        for (int i = threadId; i < width; i += numThreads)
        {
            float x = x0 + i * dx;
            float y = y0 + j * dy;

            int index = (j * width + i);
            output[index] = mandel(x, y, maxIterations);
        }
    }

    double endTime = CycleTimer::currentSeconds();
    minThread = std::min(minThread, endTime - startTime);
    printf("Thread %d run time: %f ms\n", threadId, minThread * 1000);

    return NULL;
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Multi-threading performed via pthreads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    const static int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    pthread_t workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i=0; i<numThreads; i++) {
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].output = output;
        args[i].threadId = i;
        args[i].numThreads = numThreads;
    }

    // Fire up the worker threads.  Note that numThreads-1 pthreads
    // are created and the main app thread is used as a worker as
    // well.

    for (int i=1; i<numThreads; i++)
        pthread_create(&workers[i], NULL, workerThreadStart, &args[i]);

    workerThreadStart(&args[0]);

    // wait for worker threads to complete
    for (int i=1; i<numThreads; i++)
        pthread_join(workers[i], NULL);
}
