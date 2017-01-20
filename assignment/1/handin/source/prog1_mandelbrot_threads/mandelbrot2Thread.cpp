#include <stdio.h>
#include <pthread.h>
#include <algorithm>

#include "CycleTimer.h"

typedef struct {
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    unsigned startRow;
    unsigned endRow;
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
void* worker2ThreadStart(void* threadArgs) {
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

    int startRow = args->startRow;
    int endRow = args->endRow;

    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;
    // Use Round Robin raw by raw
    for (int j = startRow; j < endRow; j++)
    {
        for (int i = 0; i < width; i++)
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
void mandelbrot2Thread(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    const static int MAX_THREADS = 32;

    pthread_t workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i = 0; i < 2; i++) {
        args[i].x0 = x0;
        args[i].x1 = x1;
        args[i].y0 = y0;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].output = output;
        args[i].threadId = i;
        args[i].numThreads = 2;
    }
    args[0].startRow = 0;
    args[0].endRow = height / 2;
    args[1].startRow = height / 2;
    args[1].endRow = height;

    // Fire up the worker threads.  Note that numThreads-1 pthreads
    // are created and the main app thread is used as a worker as
    // well.

    pthread_create(&workers[1], NULL, worker2ThreadStart, &args[1]);

    worker2ThreadStart(&args[0]);

    // wait for worker threads to complete
    pthread_join(workers[1], NULL);
}
