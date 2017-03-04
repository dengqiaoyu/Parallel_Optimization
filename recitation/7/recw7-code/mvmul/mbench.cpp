#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "CycleTimer.h"
#include "matrix.h"
#include "mbench.h"

// Home directory for benchmarks
static const char *bdir = "/afs/cs.cmu.edu/academic/class/15418-s17/recitations/recw5/benchmarks";
//static const char *bdir = "../benchmarks";

/////// Tunable parameters

// Number of benchmark files to use
static int nbench = 5;

// Number of runs to perform for each test
static int nrun = 100;

// Number of multiply functions
#define NFUNS 10

// Path template for retrieving benchmark
static const char *bpath = "%s/m-%d-%.2d-%c-%d.csr";

//// Global data structures

static struct {
    const char *name;
    mult_csr_t mult_fun;
} mult_csr_funs[NFUNS];
static index_t mult_csr_cnt = 0;

void register_mult_csr(const char *name, mult_csr_t mult_fun) {
    mult_csr_funs[mult_csr_cnt].name = name;
    mult_csr_funs[mult_csr_cnt].mult_fun = mult_fun;
    mult_csr_cnt++;
}

mult_csr_t find_mult_csr(const char *name) {
    for (index_t i = 0; i < mult_csr_cnt; i++) {
	if (strcmp(name, mult_csr_funs[i].name) == 0)
	    return mult_csr_funs[i].mult_fun;
    }
    printf("Couldn't find mult function '%s'\n", name);
    exit(1);
}

// Build pathname of benchmark file 
static void bfile_name(char *buf, int pscale, int dpct,
		       dist_t dist, int id) {
    sprintf(buf, bpath, bdir, pscale, dpct, dist_name(dist), id);
}

// Run benchmark.  Return result in gigaflops
float run_bench(int pscale, int dpct, dist_t dist, const char *mult_name) {
    char bname[256];
    double time = 0.0;
    double ops = 0.0;
    for (int id = 0; id < nbench; id++) {
	// Read in matrix
	bfile_name(bname, pscale, dpct, dist, id);
	FILE *infile = fopen(bname, "r");
	if (!infile) {
	    printf("Couldn't open benchmark file '%s'\n", bname);
	    exit(1);
	}
	csr_t *sm = read_csr(infile, bname);
	if (!sm)
	    exit(0);
	// Prepare data
	index_t nnz = sm->nnz;
	index_t nrow = sm->nrow;
	ops += (float) nnz * 2 * nrun;
	vec_t *x = new_vector(nrow);
	fill_vector(x);
	vec_t *y = new_vector(nrow); 
	// Compute expected results
	vec_t *z = new_vector(nrow);
	smvp_baseline(sm, x, z);

	// Sparse matrix benchmarking
	mult_csr_t mult_fun = find_mult_csr(mult_name);
	// Test correctness (and warm up cache)
	mult_fun(sm, x, y);
	check_vector(y, z);
	// Now do the testing
	double startTime = CycleTimer::currentSeconds();
	for (int run = 0; run < nrun; run++) {
	    mult_fun(sm, x, y);
	}
	time += CycleTimer::currentSeconds() - startTime;
	free_csr(sm);
	free_vector(x);
	free_vector(y);
	free_vector(z);
    }
    //    printf("Time = %.2f ms, Ops = %.0f\n", time * 1000.0, ops);
    return ops / (1e9 * time);
}

void run_all(int pscale, int dpct, dist_t dist, int nrunval) {
    index_t mi;
    nrun = nrunval;
    printf("\t%d\t%d\t%c\n", pscale, dpct, dist_name(dist));
    printf("\tFun\tGF\n");
    for (mi = 0; mi < mult_csr_cnt; mi++) {
	const char *mname = mult_csr_funs[mi].name;
	double gflops = run_bench(pscale, dpct, dist, mname);
	    printf("\t%s\t%.2f\n",
		   mname, gflops);
    }
}

