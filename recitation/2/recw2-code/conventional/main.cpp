#include <stdio.h>
//#include <algorithm>
#include <getopt.h>
#include <math.h>
#include <stdbool.h>
#include <pthread.h>

#include "CycleTimer.h"
#include "sin.h"

#define MAXELEMENTS 50000
#define MAXBENCH 100
#define MAXTHREADS 100

static int Nelements = 10000;
static int Nterms = 15;
static int Nreps = 1000;
static int Nthreads = 1;

static float xvals[MAXELEMENTS];
static float correct[MAXELEMENTS];
static float computed[MAXELEMENTS];


static float RELTOL = 1e-5;
static float ABSTOL = 1e-10;

benchmark_t benchmarks[MAXBENCH];

/* Version used for correctness checking */
void sinx_golden(int N, int terms, float * x, float *result);

/* Information to pass to thread */
typedef struct {
    sin_fun_t fun;
    int N;
    int reps;
    float *x;
    float *result;
} arg_t;

static void initialize(int nelements, int nterms, int nreps, int nthreads) {
    Nelements = nelements;
    Nterms = nterms;
    Nreps = nreps;
    Nthreads = nthreads;
    float delta = 1.0 / ((float) Nelements * M_2_PI);
    for (int i = 0; i < Nelements; i++) {
        xvals[i] = delta * i;
    }
    sinx_golden(Nelements, Nterms, xvals, correct);
    init_benchmarks(benchmarks);
}

static bool almost_equal(float x1, float x2) {
    float diff = fabs(x1 - x2);
    float avg = fabs(x1 + x2) / 2.0;
    return avg == 0 ? diff <= ABSTOL : diff / avg <= RELTOL;
}

void* threadfun(void *args) {
    arg_t *targs = (arg_t *) args;
    sin_fun_t fun = targs->fun;
    int N = targs->N;
    float *x = targs->x;
    float *result = targs->result;
    int reps = targs->reps;
    free(args);
    for (int r = 0; r < reps; r++)
        fun(N, Nterms, x, result);
    return NULL;
}

static void run(sin_fun_t fun, int reps, float *result) {
    pthread_t tid[MAXTHREADS];
    int npt = Nelements / Nthreads;
    for (int i = 0; i < Nthreads - 1; i++) {
        int start = i * npt;
        arg_t *args = (arg_t *) malloc(sizeof(arg_t));
        args->fun = fun;
        args->N = npt;
        args->x = xvals + start;
        args->result = result + start;
        args->reps = reps;
        pthread_create(&tid[i], NULL, threadfun, (void *) args);
    }
    /* Do the rest as the main thread */
    int start = (Nthreads - 1) * npt;
    int count = Nelements - start;
    for (int r = 0; r < reps; r++)
        fun(count, Nterms, xvals + start, result + start);

    /* Synchronize */
    for (int i = 0; i < Nthreads - 1; i++)
        pthread_join(tid[i], NULL);
}


static bool check(int i, bool threaded) {
    int badcount = 0;
    const char *name = benchmarks[i].name;
    sin_fun_t fun = benchmarks[i].fun;
    memset((void *) computed, 0, Nelements * sizeof(float));
    if (threaded)
        run(fun, 1, computed);
    else
        fun(Nelements, Nterms, xvals, computed);
    for (int i = 0; i < Nelements; i++) {
        if (!almost_equal(correct[i], computed[i])) {
            badcount++;
            if (badcount <= 3) {
                fprintf(stdout, "ERROR %s (%s): i = %d, x = %.5f, target sin(x) = %.5f, computed sin(x) = %.5f\n",
                        name, threaded ? "threaded" : "unthreaded",
                        i, xvals[i], correct[i], computed[i]);
            }
        }
    }
    if (badcount > 0) {
        fprintf(stdout, "ERROR (%s): %d/%d values computed incorrectly\n", name, badcount, Nelements);
        return false;
    }
    return true;
}


// Measure computation.  Return number of nanoseconds per element per term
static double measure(int i) {
    sin_fun_t fun = benchmarks[i].fun;
    /* Make sure cache warmed up */
    fun(Nelements, Nterms, xvals, computed);
    double startTime = CycleTimer::currentSeconds();
    run(fun, Nreps, computed);
    double endTime =  CycleTimer::currentSeconds();
    return (endTime - startTime);
}

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -n  --N     <N>    Compute for N different data values\n");
    printf("  -T  --terms <T>    Compute T terms of expansion\n");
    printf("  -r  --reps  <R>    Perform R repetitions for each value\n");
    printf("  -t  --threads <t>  Compute using t threads\n");
    printf("  -h  --help         This message\n");
}


int main(int argc, char** argv) {
    int nelements = Nelements;
    int nthreads = Nthreads;
    int nterms = Nterms;
    int nreps = Nreps;
    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"N", 1, 0, 'n'},
        {"threads", 1, 0, 't'},
        {"terms", 1, 0, 'T'},
        {"reps", 1, 0, 'r'},
        {"help", 0, 0, 'h'},
        {0 , 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "n:T:t:r:h", long_options, NULL)) != EOF) {

        switch (opt) {
        case 'n': {
            nelements = atoi(optarg);
            break;
        }
        case 't': {
            nthreads = atoi(optarg);
            break;
        }
        case 'T': {
            nterms = atoi(optarg);
            break;
        }
        case 'r': {
            nreps = atoi(optarg);
            break;
        }
        case 'h':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options
    initialize(nelements, nterms, nreps, nthreads);
    int i = 0;
    while (true) {
        const char *name = benchmarks[i].name;
        sin_fun_t fun = benchmarks[i].fun;
        if (!fun)
            break;
        if (check(i, false) && (Nthreads == 1 || check(i, true))) {
            /* Convert time to nanoseconds */
            double tsec = measure(i);
            double tmsec = tsec * 1000.0;
            double tnsec = tsec * 1e9;
            double comps = (double) (Nelements * Nterms * Nreps);
            double nspcomp = tnsec / comps;
            printf("%s: Time = %.2fms.  N = %d, T = %d,  r = %d, t = %d.  %.3f ns/element\n",
                   name, tmsec, Nelements, Nterms, Nreps, Nthreads, nspcomp);
        }
        i++;
    }
    return 0;
}

// Original version of sin function
void sinx_golden(int N, int terms, float * x, float *result) {
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
