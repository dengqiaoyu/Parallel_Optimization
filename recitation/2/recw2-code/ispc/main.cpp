#include <stdio.h>
//#include <algorithm>
#include <getopt.h>
#include <math.h>
#include <stdbool.h>

#include "CycleTimer.h"
#include "sin.h"

#include "sin_ispc.h"

#define MAXELEMENTS 50000
#define MAXBENCH 100

static int Nelements = 10000;
static int Nterms = 15;
static int Nreps = 1000;

static float xvals[MAXELEMENTS];
static float correct[MAXELEMENTS];
static volatile float computed[MAXELEMENTS];

static float RELTOL = 1e-5;
static float ABSTOL = 1e-10;

benchmark_t benchmarks[MAXBENCH];
void init_benchmarks(benchmark_t *b);

/* Version used for correctness checking */
void sinx_golden(int N, int terms, float * x, float *result);

static void initialize(int nelements, int nterms, int nreps) {
    Nelements = nelements;
    Nterms = nterms;
    Nreps = nreps;
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

static bool check(int i) {
    int badcount = 0;
    const char *name = benchmarks[i].name;
    sin_fun_t fun = benchmarks[i].fun;
    memset((void *) computed, 0, Nelements * sizeof(float));
    fun(Nelements, Nterms, xvals, (float *) computed);
    for (int i = 0; i < Nelements; i++) {
        if (!almost_equal(correct[i], computed[i])) {
            badcount++;
            if (badcount <= 5) {
                fprintf(stdout,
                        "ERROR (%s): i = %d, x = %.5f, target sin(x) = %.5f, computed sin(x) = %.5f\n",
                        name, i, xvals[i], correct[i], computed[i]);
            }
        }
    }
    if (badcount > 0) {
        fprintf(stdout,
                "ERROR (%s): %d/%d values computed incorrectly\n",
                name, badcount, Nelements);
        return false;
    }
    return true;
}

// Measure computation.  Return time in seconds
static double measure(int i) {
    sin_fun_t fun = benchmarks[i].fun;
    /* Make sure cache warmed up */
    fun(Nelements, Nterms, xvals, (float *) computed);
    double startTime = CycleTimer::currentSeconds();
    for (int r = 0; r < Nreps; r++) {
        fun(Nelements, Nterms, xvals, (float *) computed);
    }
    double endTime =  CycleTimer::currentSeconds();
    return (endTime - startTime);
}

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -n  --N     <N>  Compute for N different data values\n");
    printf("  -t  --terms <T>  Compute T terms of expansion\n");
    printf("  -r  --reps  <R>  Perform R repetitions for each value\n");
    printf("  -h  --help       This message\n");
}


int main(int argc, char** argv) {
    int nelements = Nelements;
    int nterms = Nterms;
    int nreps = Nreps;
    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"N", 1, 0, 'n'},
        {"terms", 1, 0, 't'},
        {"reps", 1, 0, 'r'},
        {"help", 0, 0, 'h'},
        {0 , 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "n:t:r:h", long_options, NULL))
            != EOF) {

        switch (opt) {
        case 'n': {
            nelements = atoi(optarg);
            break;
        }
        case 't': {
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
    initialize(nelements, nterms, nreps);
    int i = 0;
    while (true) {
        const char *name = benchmarks[i].name;
        sin_fun_t fun = benchmarks[i].fun;
        if (!fun)
            break;
        if (check(i)) {
            /* Convert time to nanoseconds */
            double tsec = measure(i);
            double tmsec = tsec * 1000.0;
            double tnsec = tsec * 1e9;
            double comps = (double) (Nelements * Nterms * Nreps);
            double nspcomp = tnsec / comps;

            printf("%s: T = %.2fms.  N = %d, t = %d,  r = %d.  %.3f ns/ele\n",
                   name, tmsec, Nelements, Nterms, Nreps, nspcomp);
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

void init_benchmarks(benchmark_t *b) {
    int i = 0;
    b[i].name = "reference"; b[i++].fun = ispc::sinx_reference;
    b[i].name = "best";    b[i++].fun = ispc::sinx_best;
    b[i].name = "ref-t2"; b[i++].fun = ispc::sinx_reference_task2;
    b[i].name = "best-t2"; b[i++].fun = ispc::sinx_best_task2;
    b[i].name = "";          b[i++].fun = NULL;
}

