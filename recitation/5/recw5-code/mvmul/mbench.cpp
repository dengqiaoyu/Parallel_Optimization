#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "CycleTimer.h"
#include "matrix.h"
#include "mbench.h"

// Home directory for benchmarks
static const char *bdir = "/afs/cs.cmu.edu/academic/class/15418-s17/recitations/recw5/benchmarks";

/////// Tunable parameters

// Number of benchmark files to use
static int nbench = 5;

// Number of runs to perform for each test
static int nrun = 100;

// Number of functions of each type
#define NFUNS 10

// Path template for retrieving benchmark
static const char *bpath = "%s/m-%d-%.2d-%c-%d.csr";

//// Global data structures

static struct {
    const char *name;
    rvp_csr_t rp_fun;
} rvp_csr_funs[NFUNS];
static index_t rvp_csr_cnt = 0;

static struct {
    const char *name;
    mvp_csr_t mp_fun;
} mvp_csr_funs[NFUNS];
static index_t mvp_csr_cnt = 0;

static struct {
    const char *name;
    rvp_dense_t rp_fun;
} rvp_dense_funs[NFUNS];
static index_t rvp_dense_cnt = 0;

static struct {
    const char *name;
    mvp_dense_t mp_fun;
} mvp_dense_funs[NFUNS];
static index_t mvp_dense_cnt = 0;

void register_rvp_csr(const char *name, rvp_csr_t rp_fun) {
    rvp_csr_funs[rvp_csr_cnt].name = name;
    rvp_csr_funs[rvp_csr_cnt].rp_fun = rp_fun;
    rvp_csr_cnt++;
}

void register_mvp_csr(const char *name, mvp_csr_t mp_fun) {
    mvp_csr_funs[mvp_csr_cnt].name = name;
    mvp_csr_funs[mvp_csr_cnt].mp_fun = mp_fun;
    mvp_csr_cnt++;
}

void register_rvp_dense(const char *name, rvp_dense_t rp_fun) {
    rvp_dense_funs[rvp_dense_cnt].name = name;
    rvp_dense_funs[rvp_dense_cnt].rp_fun = rp_fun;
    rvp_dense_cnt++;
}

void register_mvp_dense(const char *name, mvp_dense_t mp_fun) {
    mvp_dense_funs[mvp_dense_cnt].name = name;
    mvp_dense_funs[mvp_dense_cnt].mp_fun = mp_fun;
    mvp_dense_cnt++;
}

rvp_csr_t find_rvp_csr(const char *name) {
    for (index_t i = 0; i < rvp_csr_cnt; i++) {
	if (strcmp(name, rvp_csr_funs[i].name) == 0)
	    return rvp_csr_funs[i].rp_fun;
    }
    printf("Couldn't find rvp function '%s'\n", name);
    exit(1);
}

mvp_csr_t find_mvp_csr(const char *name) {
    for (index_t i = 0; i < mvp_csr_cnt; i++) {
	if (strcmp(name, mvp_csr_funs[i].name) == 0)
	    return mvp_csr_funs[i].mp_fun;
    }
    printf("Couldn't find mvp function '%s'\n", name);
    exit(1);
}

rvp_dense_t find_rvp_dense(const char *name) {
    for (index_t i = 0; i < rvp_dense_cnt; i++) {
	if (strcmp(name, rvp_dense_funs[i].name) == 0)
	    return rvp_dense_funs[i].rp_fun;
    }
    printf("Couldn't find rvp function '%s'\n", name);
    exit(1);
}

mvp_dense_t find_mvp_dense(const char *name) {
    for (index_t i = 0; i < mvp_dense_cnt; i++) {
	if (strcmp(name, mvp_dense_funs[i].name) == 0)
	    return mvp_dense_funs[i].mp_fun;
    }
    printf("Couldn't find mvp function '%s'\n", name);
    exit(1);
}

// Build pathname of benchmark file 
static void bfile_name(char *buf, int pscale, int dpct,
		       dist_t dist, int id) {
    sprintf(buf, bpath, bdir, pscale, dpct, dist_name(dist), id);
}

// Run benchmark.  Return result in gigaflops
float run_bench(int pscale, int dpct, dist_t dist, bool densify,
	        const char *rvp_name, const char *mvp_name) {
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
	csr_t *sm = read_csr(infile);
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

	if (densify) {
	    // Dense matrix benchmarking
	    rvp_dense_t rp_fun = find_rvp_dense(rvp_name);
	    mvp_dense_t mp_fun = find_mvp_dense(mvp_name);
	    dense_t *dm = csr_to_dense(sm);
	    // Test correctness (and warm up cache)
	    dmvp(dm, x, y, mp_fun, rp_fun);
	    check_vector(y, z);
	    // Now do the testing
	    double startTime = CycleTimer::currentSeconds();
	    for (int run = 0; run < nrun; run++) {
		dmvp(dm, x, y, mp_fun, rp_fun);
	    }
	    time += CycleTimer::currentSeconds() - startTime;
	    free_dense(dm);
	} else {
	    // Sparse matrix benchmarking
	    rvp_csr_t rp_fun = find_rvp_csr(rvp_name);
	    mvp_csr_t mp_fun = find_mvp_csr(mvp_name);
	    // Test correctness (and warm up cache)
	    smvp(sm, x, y, mp_fun, rp_fun);
	    check_vector(y, z);
	    // Now do the testing
	    double startTime = CycleTimer::currentSeconds();
	    for (int run = 0; run < nrun; run++) {
		smvp(sm, x, y, mp_fun, rp_fun);
	    }
	    time += CycleTimer::currentSeconds() - startTime;
	}
	free_csr(sm);
	free_vector(x);
	free_vector(y);
	free_vector(z);
    }
    //    printf("Time = %.2f ms, Ops = %.0f\n", time * 1000.0, ops);
    return ops / (1e9 * time);
}

void run_all_sparse(int pscale, int dpct, dist_t dist) {
    index_t ri, mi;
    printf("Sparse\t%d\t%d\t%c\n", pscale, dpct, dist_name(dist));
    printf("\tMVP\tRVP\tGF\n");
    for (mi = 0; mi < mvp_csr_cnt; mi++) {
	const char *mname = mvp_csr_funs[mi].name;
	for (ri = 0; ri < rvp_csr_cnt; ri++) {
	    const char *rname = rvp_csr_funs[ri].name;
	    double gflops = run_bench(pscale, dpct, dist, false,
				      rname, mname);
	    printf("\t%s\t%s\t%.2f\n",
		   mname, rname, gflops);
	}
    }
}

void run_all_dense(int pscale, int dpct, dist_t dist) {
    index_t ri, mi;
    printf("Dense\t%d\t%d\t%c\n", pscale, dpct, dist_name(dist));
    printf("\tMVP\tRVP\tGF\n");
    for (mi = 0; mi < mvp_dense_cnt; mi++) {
	const char *mname = mvp_dense_funs[mi].name;
	for (ri = 0; ri < rvp_dense_cnt; ri++) {
	    const char *rname = rvp_dense_funs[ri].name;
	    double gflops = run_bench(pscale, dpct, dist, true,
				      rname, mname);
	    printf("\t%s\t%s\t%.2f\n",
		   mname, rname, gflops);
	}
    }
}

