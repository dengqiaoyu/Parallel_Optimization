// Library of row multiplication and matrix multiplication functions
#include <string.h>
#include <stdbool.h>

#include "matrix.h"
#include "mbench.h"

///////////////// SPARSE MATRICES - ROW-BASED /////////////////////////

// Standard sequential implementation
void mul_csr_seq(csr_t *m, vec_t *x, vec_t *y) {
    index_t nrow = m->nrow;
    for (index_t r = 0; r < nrow; r++) {
	index_t idxmin = m->rowstart[r];
	index_t idxmax = m->rowstart[r+1];
	float val = 0.0;
	for (index_t idx = idxmin; idx < idxmax; idx++) {
	    index_t c = m->cindex[idx];
	    data_t mval = m->value[idx];
	    data_t xval = x->value[c];
	    val += mval * xval;
	}
	y->value[r] = val;
    }
}

// Using OMP static parallel over rows
void mul_csr_mprow(csr_t *m, vec_t *x, vec_t *y) {
    index_t nrow = m->nrow;
    #pragma omp parallel for schedule(static)
    for (index_t r = 0; r < nrow; r++) {
	index_t idxmin = m->rowstart[r];
	index_t idxmax = m->rowstart[r+1];
	float val = 0.0;
	for (index_t idx = idxmin; idx < idxmax; idx++) {
	    index_t c = m->cindex[idx];
	    data_t mval = m->value[idx];
	    data_t xval = x->value[c];
	    val += mval * xval;
	}
	y->value[r] = val;
    }
}

///////////////// SPARSE MATRICES - CONTENTS-BASED /////////////

// Sequential over elements, updates to memory
void mul_csr_data_seq(csr_t *m, vec_t *x, vec_t *y) {
    clear_vector(y);
    for (index_t idx = 0; idx < m->nnz; idx++) {
	index_t r = m->rindex[idx];
	index_t c = m->cindex[idx];
	data_t xval = x->value[c];
	data_t mval = m->value[idx];
	y->value[r] += mval * xval;
    }
}

// Parallel over elements, updates to memory 
void mul_csr_data_mps(csr_t *m, vec_t *x, vec_t *y) {
    clear_vector(y);
    #pragma omp parallel for schedule(static)
    for (index_t idx = 0; idx < m->nnz; idx++) {
	index_t r = m->rindex[idx];
	index_t c = m->cindex[idx];
	data_t xval = x->value[c];
	data_t mval = m->value[idx];
        #pragma omp atomic
	y->value[r] += mval * xval;
    }
}

// Sequential over elements, updates to registers
void mul_csr_rdata_seq(csr_t *m, vec_t *x, vec_t *y) {
    clear_vector(y);
    index_t last_r = 0;
    data_t val = 0.0;
    for (index_t idx = 0; idx < m->nnz; idx++) {
	index_t r = m->rindex[idx];
	if (r != last_r) {
	    if (val != 0.0)
		y->value[last_r] += val;
	    last_r = r;
	    val = 0.0;
	}
	index_t c = m->cindex[idx];
	data_t xval = x->value[c];
	data_t mval = m->value[idx];
	val += mval * xval;
    }
    if (val != 0.0)
	y->value[last_r] += val;
}

// Parallel over elements, updates to registers
// Shared destination vector
void mul_csr_rdata_mps(csr_t *m, vec_t *x, vec_t *y) {
    clear_vector(y);
    #pragma omp parallel
    {
	index_t last_r = 0;  // Private to thread
	data_t val = 0.0;    // Private to thread
        #pragma omp for schedule(static)
	for (index_t idx = 0; idx < m->nnz; idx++) {
	    index_t r = m->rindex[idx];
	    if (r != last_r) {
		if (val != 0.0)
                    #pragma omp atomic
		    y->value[last_r] += val;
		last_r = r;
		val = 0.0;
	    }
	    index_t c = m->cindex[idx];
	    data_t xval = x->value[c];
	    data_t mval = m->value[idx];
	    val += mval * xval;
	}
	if (val != 0.0)
            #pragma omp atomic
	    y->value[last_r] += val;
    }
}

// Parallel over elements, updates to registers
// Separate destination vector
void mul_csr_rldata_mps(csr_t *m, vec_t *x, vec_t *y) {
    clear_vector(y);
    #pragma omp parallel
    {
        index_t nrow = m->nrow;
	index_t last_r = 0;           // Private to thread
	data_t val = 0.0;             // Private to thread
	data_t local_y[nrow];         // Private to thread
	index_t min_r = 0;            // Private to thread
	bool first = true;            // Private to thread
	memset((void *) local_y, 0, nrow * sizeof(data_t));
        #pragma omp for schedule(static) nowait
	for (index_t idx = 0; idx < m->nnz; idx++) {
	    index_t r = m->rindex[idx];
	    if (first)
		min_r = last_r = r;
	    first = false;
	    if (r != last_r) {
	        local_y[last_r] += val;
		last_r = r;
		val = 0.0;
	    }
	    index_t c = m->cindex[idx];
	    data_t xval = x->value[c];
	    data_t mval = m->value[idx];
	    val += mval * xval;
	}
	local_y[last_r] += val;
	// Combine local y values
	for (index_t r = min_r; r <= last_r; r++) {
	    #pragma omp atomic
	    y->value[r] += local_y[r];
	}
    }
}


///////////////// REGISTRATION/////////////////////////

// Registration of all functions
void register_functions() {
    register_mult_csr("seq-row", mul_csr_seq);
    register_mult_csr("mps-row", mul_csr_mprow);
    register_mult_csr("seq-mdt", mul_csr_data_seq);
    register_mult_csr("mps-mdt", mul_csr_data_mps);
    register_mult_csr("seq-rdt", mul_csr_rdata_seq);
    register_mult_csr("mps-rdt", mul_csr_rdata_mps);
    register_mult_csr("mps-rld", mul_csr_rldata_mps);
}

