/* Functions for implementing sparse matrix operations */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "matrix.h"

////////////////// UTILITIES ////////////////////////////

// Functions to aid with random matrix/vector generation
void set_seed(unsigned u) {
    srandom(u+1);
}

// Names
static char dname[3] = { 'u', 'r', 's'};

dist_t find_dist(char dc) {
    int i;
    for (i = 0; i < DIST_BAD; i++)
	if (dname[i] == dc)
	    return (dist_t) i;
    return DIST_BAD;
}

char dist_name(dist_t dist) {
    if (dist < DIST_BAD)
	return dname[dist];
    return '?';
}



////////////////// VECTORS ////////////////////////////

vec_t *new_vector(index_t length) {
    vec_t *v = (vec_t *) malloc(sizeof(vec_t));
    v->length = length;
    v->value = (data_t *) calloc(length, sizeof(data_t));
    return v;
}

void free_vector(vec_t *vec) {
    free(vec->value);
    free(vec);
}

void clear_vector(vec_t *vec) {
    memset(vec->value, 0, vec->length * sizeof(data_t));
}

void fill_vector(vec_t *vec) {
    index_t i;
    for (i = 0; i < vec->length; i++)
	vec->value[i] = (float) random()/RAND_MAX;
}

static float epsilon = 0.001;
static bool almost_equal(data_t x, data_t y) {
    data_t diff = fabs(x-y);
    return diff <= epsilon;
}

bool check_vector(vec_t *x, vec_t *y) {
    index_t i;
    for (i = 0; i < x->length; i++) {
	if (!almost_equal(x->value[i], y->value[i])) {
	    printf("Mismatch at element %u.  %.5f != %.5f\n",
		   i, x->value[i], y->value[i]);
	    return false;
	}
    }
    return true;
}

void show_vector(vec_t *vec) {
    index_t i;
    printf("[");
    for (i = 0; i < vec->length; i++) {
	printf("\t%.2f", vec->value[i]);
    }
    printf("]\n");
}

////////////////// SPARSE MATRICES ////////////////////////////

csr_t *new_csr(index_t nrow, index_t nnz) {
    csr_t *m = (csr_t *) malloc(sizeof(csr_t));
    m->nrow = nrow;
    m->nnz = nnz;
    m->value = (data_t *) calloc(nnz, sizeof(data_t));
    m->cindex = (index_t *) calloc(nnz, sizeof(index_t));
    m->rindex = (index_t *) calloc(nnz, sizeof(index_t));
    m->rowstart = (index_t *) calloc(nrow+1, sizeof(index_t));
    return m;
}

void free_csr(csr_t *m) {
    free(m->value);
    free(m->cindex);
    free(m->rindex);
    free(m->rowstart);
    free(m);
}

int icomp(const void *ip, const void *jp) {
    index_t i = *(index_t *) ip;
    index_t j = *(index_t *) jp;
    if (i == j)
	return 0;
    if (i < j)
	return -1;
    return 1;
}


// Select cnt elements from set of values 0 ... range-1
// Parameter set must be large enough to hold cnt values
static void select_set(index_t *set, index_t range, index_t cnt) {
    index_t i;
    index_t buf[range];
    for (i = 0; i < range; i++) {
	buf[i] = i;
    }
    for (i = 0; i < cnt; i++) {
	float u = (float) random()/RAND_MAX;
	index_t idx = (range-i-1) * u + i;
	if (idx >= range) {
	    printf("Warning.  select_set tried to use index %u\n", idx);
	    printf("\trange = %d, cnt = %d, i = %u, u = %.4f\n", range, cnt, i, u);
	}
	index_t t = buf[i];
	buf[i] = buf[idx];
	buf[idx] = t;
    }
    qsort((void *) buf, cnt, sizeof(index_t), icomp);
    for (i = 0; i < cnt; i++) {
	if (buf[i] >= range) {
	    printf("Warning.  select_set got out of range value %u\n", buf[i]);
	    printf("\trange = %d, cnt = %d\n", range, cnt);
	}
	set[i] = buf[i];
    }
}

// Check for any discrepancies in a CSR
bool check_csr(csr_t *m, const char *name) {
    index_t nrow = m->nrow;
    index_t nnz = m->nnz;
    index_t r;
    index_t c;
    index_t idx = 0;
    index_t clast = 0;
    for (r = 0; r < nrow; r++) {
	clast = 0;
	if (idx != m->rowstart[r]) {
	    printf("Incorrect CSR '%s'.  Rowstart[%u] = %u.  Should be %u\n",
		   name, r, m->rowstart[r], idx);
	    return false;
	}
	while (idx < m->rowstart[r+1]) {
	    if (r != m->rindex[idx]) {
		printf("Incorrect CSR '%s'.  Row %u.  Index %u.  Incorrect row index %u\n",
		       name, r, idx, m->rindex[idx]);
		return false;
	    }
	    c = m->cindex[idx];
	    if (c >= nrow) {
		printf("Incorrect CSR '%s'.  Row %u.  Invalid column number %u\n",
		       name, r, c);
		return false;
	    }
	    if (idx > m->rowstart[r] && c <= clast) {
		printf("Incorrect CSR '%s'.  Row %u.  Invalid column sequence %u, %u\n",
		       name, r, clast, c);
		return false;
	    }
	    clast = c;
	    idx++;
	}
    }
    // Make sure number of nonzeros is correct:
    if (nnz != m->rowstart[nrow]) {
	printf("Incorrect CSR '%s'.  Total nonzeros = %u.  Last rowstart = %u\n",
	       name, nnz, m->rowstart[nrow]);
	return false;
    }
    return true;
}

// Partition rows according to target density and distribution type
static void assign_rowcounts(index_t *counts, index_t nrow,
			     index_t nnz, dist_t dist) {
    index_t r;
    if (dist == DIST_UNIFORM) {
	index_t nleft = nnz;
	for (r = 0; r < nrow; r++) {
	    index_t cnt = (nleft + nrow - r - 1)/(nrow - r);
	    counts[r] = cnt;
	    nleft -= cnt;
	}
    } else if (dist == DIST_SKEW) {
	index_t nleft = nnz;
	for (r = 0; r < nrow; r++) {
	    index_t cnt;
	    if (nleft >= nrow)
		// Fill first rows completely full
		cnt = nrow;
	    else
		// Fill rest uniformly
		cnt = (nleft + nrow - r - 1) / (nrow - r);
	    counts[r] = cnt;
	    nleft -= cnt;
	}
    } else {
	// Create vector of random values between 0 and 1
	float rvec[nrow];
	float sum = 0.0;
	for (r = 0; r < nrow; r++) {
	    rvec[r] = (float) random() / RAND_MAX;
	    sum += rvec[r];
	}

	index_t nleft = nnz;
	// Now use these numbers to compute initial row counts
	for (r = 0; r < nrow; r++) {
	    index_t cnt = (index_t) (rvec[r]/sum * nnz);
	    if (cnt > nrow)
		cnt = nrow;
	    counts[r] = cnt;
	    nleft -= cnt;
	}

	// Fill rows with any leftover 
	while (nleft > 0) {
	    for (r = 0; r < nrow && nleft > 0; r++) {
		if (counts[r] < nrow) {
		    counts[r]++;
		    nleft--;
		}
	    }
	}
    }
}


// Generate matrix with specified density, and with row/column sizes
// having different distribution characteristics
csr_t *gen_csr(index_t nrow, float density, dist_t dist) {
    index_t nnz = (index_t) roundf(density * nrow * nrow);
    csr_t *m = new_csr(nrow, nnz);
    index_t counts[nrow];
    assign_rowcounts(counts, nrow, nnz, dist);
    index_t idx = 0;
    for (index_t r = 0; r < nrow; r++) {
	m->rowstart[r] = idx;
	index_t tcol = counts[r];
	select_set(&m->cindex[idx], nrow, tcol);
	while (idx < m->rowstart[r] + tcol) {
		float nval = 2.0 * ((float) random()/RAND_MAX - 0.5);
		m->rindex[idx] = r;
		m->value[idx++] = nval;

	}
    }
    m->rowstart[nrow] = nnz;
    if (!check_csr(m, "Generated")) {
	printf("Failed to produce CSR with %u rows, density = %.2f, type = %c\n",
	       nrow, density, dname[dist]);
	return NULL;
    }
    return m;
}

void show_csr(csr_t *m) {
    index_t r, c;
    for (r = 0; r < m->nrow; r++) {
	index_t rsize = 0;
	index_t idx = m->rowstart[r];
	for (c = 0; c < m->nrow; c++) {
	    if (idx < m->rowstart[r+1] && m->cindex[idx] == c) {
		printf("\t%.2f", m->value[idx]);
		idx++;
		rsize++;
	    } else
		printf("\t0.00");
	}
	printf("\t(%u)\n", rsize);
    }
}

// Save CSR Matrix to file
void save_csr(csr_t *m, FILE *outfile) {
    index_t nrow = m->nrow;
    index_t nnz = m->nnz;
    fprintf(outfile, "%u %u\n", nrow, nnz);
    index_t idx, r;
    for (r = 0; r <= nrow; r++)
	fprintf(outfile, "%u\n", m->rowstart[r]);
    for (idx = 0; idx < nnz; idx++)
	fprintf(outfile, "%u %f\n", m->cindex[idx], m->value[idx]);
    fclose(outfile);
}

// Error checking for scanf functions
static void check_scan(int eval, int rval) {
    if (rval != eval) {
	printf("Error reading with scanf.  Expected %d values.  Got %d\n",
	       eval, rval);
    }
}

// Load CSR Matrix from file
csr_t *read_csr(FILE *infile, const char *fname) {
    index_t nrow, nnz;
    check_scan(2, fscanf(infile, "%u %u\n", &nrow, &nnz));
    csr_t *m = new_csr(nrow, nnz);
    index_t r, idx;
    for (r = 0; r <= nrow; r++) {
	check_scan(1, fscanf(infile, "%u\n", &(m->rowstart[r])));
    }
    r = 0;
    for (idx = 0; idx < nnz; idx++) {
	while (idx == m->rowstart[r+1])
	    r++;
	check_scan(2, fscanf(infile, "%u %f\n",
			     &(m->cindex[idx]), &(m->value[idx])));
	m->rindex[idx] = r;
    }
    if (!check_csr(m, fname))
	return NULL;
    return m;
}

float rvp_csr_baseline(csr_t *m, vec_t *x, index_t r) {
    index_t idxmin = m->rowstart[r];
    index_t idxmax = m->rowstart[r+1];
    index_t idx;
    float val = 0.0;
    for (idx = idxmin; idx < idxmax; idx++) {
	index_t c = m->cindex[idx];
	data_t mval = m->value[idx];
	data_t xval = x->value[c];
	val += mval * xval;
    }
    return val;
}

// Baseline implementation of matrix product function

void smvp_baseline(csr_t *m, vec_t *x, vec_t *y) {
    index_t nrow = m->nrow;
    index_t r;
    for (r = 0; r < nrow; r++) {
	y->value[r] = rvp_csr_baseline(m, x, r);
    }
}
