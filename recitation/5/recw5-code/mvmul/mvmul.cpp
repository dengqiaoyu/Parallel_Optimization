// Library of row multiplication and matrix multiplication functions
#include "matrix.h"
#include "mbench.h"

/// Standard level of unrolling
#define UNROLL 3

/// Vectorizing for AVX2
#define VBYTES 32
#define VSIZE (VBYTES/sizeof(data_t))

typedef data_t vector_t __attribute__((vector_size(VBYTES)));

///////////////// SPARSE ROWS ////////////////////////////


// Standard sequential version
float rvp_csr_seq(csr_t *m, vec_t *x, index_t r) {
    index_t idxmin = m->rowstart[r];
    index_t idxmax = m->rowstart[r + 1];
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

// Use unrolling and multiple accumulators
float rvp_csr_par(csr_t *m, vec_t *x, index_t r) {
    index_t idxmin = m->rowstart[r];
    index_t idxmax = m->rowstart[r + 1];
    index_t idx;
    float uval[UNROLL];
    float val = 0.0;
    for (index_t i = 0; i < UNROLL; i++)
        uval[i] = 0.0;
    for (idx = idxmin; idx + UNROLL <= idxmax; idx += UNROLL) {
        for (index_t i = 0; i < UNROLL; i++) {
            index_t c = m->cindex[idx + i];
            data_t mval = m->value[idx + i];
            data_t xval = x->value[c];
            uval[i] += mval * xval;
        }
    }
    for (; idx < idxmax; idx ++) {
        index_t c = m->cindex[idx];
        data_t mval = m->value[idx];
        data_t xval = x->value[c];
        val += mval * xval;
    }

    for (index_t i = 0; i < UNROLL; i++)
        val += uval[i];
    return val;
}


// OMP Reduction
float rvp_csr_mpr(csr_t *m, vec_t *x, index_t r) {
    index_t idxmin = m->rowstart[r];
    index_t idxmax = m->rowstart[r + 1];
    index_t idx;
    float val = 0.0;
    // Pragma to do parallel reduction on val

    // Your code here

    #pragma omp parallel for reduction(+:val)
    for (idx = idxmin; idx < idxmax; idx++) {
        index_t c = m->cindex[idx];
        data_t mval = m->value[idx];
        data_t xval = x->value[c];
        val += mval * xval;
    }
    return val;
}

// Use explicit unrolling and multiple accumulators
float rvp_csr_mpr3(csr_t *m, vec_t *x, index_t r) {
    index_t idxmin = m->rowstart[r];
    index_t idxmax = m->rowstart[r + 1];
    index_t idx = idxmin;
    float val0 = 0.0;
    float val1 = 0.0;
    float val2 = 0.0;
    float val = 0.0;
    index_t cnt = (idxmax - idxmin) / 3;
    // Pragma to do parallel reduction on val0, val1, and val2

    // Your code here
    #pragma omp parallel for reduction(+:val0, val1, val2)
    for (index_t i = 0; i < cnt; i++) {
        val0 += m->value[idx + 3 * i + 0] * x->value[m->cindex[idx + 3 * i + 0]];
        val1 += m->value[idx + 3 * i + 1] * x->value[m->cindex[idx + 3 * i + 1]];
        val2 += m->value[idx + 3 * i + 2] * x->value[m->cindex[idx + 3 * i + 2]];
    }
    idx += 3 * cnt;
    for (; idx < idxmax; idx ++) {
        index_t c = m->cindex[idx];
        data_t mval = m->value[idx];
        data_t xval = x->value[c];
        val += mval * xval;
    }

    val += val0 + (val1 + val2);
    return val;
}


///////////////// SPARSE MATRICES /////////////////////////

// Baseline implementation of matrix product function
void mvp_csr_seq(csr_t *m, vec_t *x, vec_t *y, rvp_csr_t rp_fun) {
    index_t nrow = m->nrow;
    index_t r;
    for (r = 0; r < nrow; r++) {
        y->value[r] = rp_fun(m, x, r);
    }
}

// Version using OMP static
void mvp_csr_mps(csr_t *m, vec_t *x, vec_t *y, rvp_csr_t rp_fun) {
    index_t nrow = m->nrow;
    index_t r;
    // pragma to parallelize loop statically

    //qdeng edited
    #pragma omp parallel for schedule(static)
    for (r = 0; r < nrow; r++) {
        y->value[r] = rp_fun(m, x, r);
    }
}

// Version using OMP dynamic
void mvp_csr_mpd(csr_t *m, vec_t *x, vec_t *y, rvp_csr_t rp_fun) {
    index_t nrow = m->nrow;
    index_t r;
    // pragma to parallelize loop dynamically

    // Your code here
    #pragma omp parallel for schedule(dynamic)
    for (r = 0; r < nrow; r++) {
        y->value[r] = rp_fun(m, x, r);
    }
}


///////////////// DENSE ROWS ////////////////////////////

// Baseline implementation of dense row product function
float rvp_dense_seq(dense_t *m, vec_t *x, index_t r) {
    index_t nrow = m->nrow;
    index_t c;
    index_t idx = r * nrow;
    float val = 0.0;
    for (c = 0; c < nrow; c++)
        val += x->value[c] * m->value[idx++];
    return val;
}

// Parallel accumulation implementation of dense row product function
float rvp_dense_par(dense_t *m, vec_t *x, index_t r) {
    index_t nrow = m->nrow;
    index_t c;
    index_t idx = r * nrow;
    float val = 0.0;
    float uval[UNROLL];
    for (index_t i = 0; i < UNROLL; i++)
        uval[i] = 0.0;
    for (c = 0; c + UNROLL <= nrow; c += UNROLL) {
        for (index_t i = 0; i < UNROLL; i++)
            uval[i] += x->value[c + i] * m->value[idx + i];
        idx += UNROLL;
    }
    for (; c < nrow; c++)
        val += x->value[c] * m->value[idx++];
    for (index_t i = 0; i < UNROLL; i++)
        val += uval[i];
    return val;
}

// Use gcc vector extensions
float rvp_dense_vec(dense_t *m, vec_t *x, index_t r) {
    index_t nrow = m->nrow;
    index_t c;
    index_t idx = r * nrow;
    vector_t vval;
    for (index_t j = 0; j < VSIZE; j++)
        vval[j] = 0.0;
    for (c = 0; c + VSIZE <= nrow; c += VSIZE) {
        vector_t *xv = (vector_t *) &x->value[c];
        vector_t *mv = (vector_t *) &m->value[idx];
        vval += *xv * *mv;
        idx += VSIZE;
    }
    float val = 0.0;
    for (index_t j = 0; j < VSIZE; j++)
        val += vval[j];
    for (; c < nrow; c++)
        val += x->value[c] * m->value[idx++];
    return val;
}

// Parallel accumulation implementation of dense row product function
// with vector extension
float rvp_dense_pvec(dense_t *m, vec_t *x, index_t r) {
    index_t nrow = m->nrow;
    index_t c;
    index_t idx = r * nrow;
    float val = 0.0;
    vector_t vuval[UNROLL];
    for (index_t i = 0; i < UNROLL; i++)
        for (index_t j = 0; j < VSIZE; j++)
            vuval[i][j] = 0.0;
    // Unrolled and vectorized
    for (c = 0; c + VSIZE * UNROLL <= nrow; c += VSIZE * UNROLL) {
        for (index_t i = 0; i < UNROLL; i++) {
            vector_t *xv = (vector_t *) &x->value[c + i * VSIZE];
            vector_t *mv = (vector_t *) &m->value[idx + i * VSIZE];
            vuval[i] += *xv * *mv;
        }
        idx += VSIZE * UNROLL;
    }
    // Vectorized
    for (; c + VSIZE <= nrow; c += VSIZE) {
        vector_t *xv = (vector_t *) &x->value[c];
        vector_t *mv = (vector_t *) &m->value[idx];
        vuval[0] += *xv * *mv;
        idx += VSIZE;
    }
    for (; c < nrow; c++)
        val += x->value[c] * m->value[idx++];
    for (index_t i = 1; i < UNROLL; i++)
        vuval[0] += vuval[i];
    for (index_t j = 0; j < VSIZE; j++)
        val += vuval[0][j];
    return val;
}


///////////////// DENSE MATRICES/////////////////////////

// Baseline implementation of dense matrix product function
void mvp_dense_seq(dense_t *m, vec_t *x, vec_t *y, rvp_dense_t rp_fun) {
    index_t nrow = m->nrow;
    index_t r;
    for (r = 0; r < nrow; r++) {
        y->value[r] = rp_fun(m, x, r);
    }
}

// Dense matrix product function with static scheduling
void mvp_dense_mps(dense_t *m, vec_t *x, vec_t *y, rvp_dense_t rp_fun) {
    index_t nrow = m->nrow;
    index_t r;
    // pragma to parallelize loop statically
    #pragma omp parallel for schedule(static)
    for (r = 0; r < nrow; r++) {
        y->value[r] = rp_fun(m, x, r);
    }
}

///////////////// REGISTRATION/////////////////////////

// Registration of all functions
void register_functions() {
    // Sparse rows
    register_rvp_csr("seq", rvp_csr_seq);
    register_rvp_csr("par", rvp_csr_par);
    register_rvp_csr("mpr", rvp_csr_mpr);
    register_rvp_csr("mpr3", rvp_csr_mpr3);

    // Sparse matrices
    register_mvp_csr("seq", mvp_csr_seq);
    register_mvp_csr("mps", mvp_csr_mps);
    register_mvp_csr("mpd", mvp_csr_mpd);

    // Dense rows
    register_rvp_dense("seq", rvp_dense_seq);
    register_rvp_dense("par", rvp_dense_par);
    register_rvp_dense("vec", rvp_dense_vec);
    register_rvp_dense("pvec", rvp_dense_pvec);

    // Dense matrices
    register_mvp_dense("seq", mvp_dense_seq);
    register_mvp_dense("mps", mvp_dense_mps);
}

