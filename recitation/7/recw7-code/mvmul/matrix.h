/* Compressed-row representation of sparse matrices */
#include <stdio.h>
#include <stdbool.h>

////////////////// TYPEDEFS ////////////////////////////

/////////// GENERAL 

// Underlying data type for representing matrix indices.  Making it
// 32-bits will speed up the program at the cost of scalability.
typedef unsigned index_t;

// Type of matrix data
typedef float data_t;

// Characteristic distributions of row sizes
typedef enum { DIST_UNIFORM, DIST_RANDOM, DIST_SKEW, DIST_BAD } dist_t;

dist_t find_dist(char dc);

char dist_name(dist_t dist);
/* insert_break */

// Required memory alignment for vector and matrix data
#define ALIGN 32

/////////// VECTORS

// Dense representation of vector
typedef struct {
    index_t length;
    // Value accessed
    data_t *value;
} vec_t;

/////////// SPARSE MATRICES

// CSR representation of square matrix
typedef struct {
    index_t nrow;       // Number of rows (= number of columns)
    index_t nnz;        // Number of nonzero elements
    data_t *value;      // Nonzero matrix values, row-major order [nnz]
    index_t *cindex;    // Column index for each nonzero entry    [nnz]
    index_t *rindex;    // Row index for each nonzero entry       [nnz]
    index_t *rowstart;  // Offset of each row                     [nrow+1]
} csr_t;

/////////// FUNCTION TYPES

// Matrix-vector multiplication function.  Compute y = Mx
typedef void (*mult_csr_t)(csr_t *m, vec_t *x, vec_t *y);


////////////////// UTILITIES ////////////////////////////

// Set random number seed
void set_seed(unsigned u);

////////////////// VECTORS ////////////////////////////

// Allocate all storage for vector
vec_t *new_vector(index_t length);

// Free all storage for a vector
void free_vector(vec_t *vec);

// Set all entries to zero
void clear_vector(vec_t *vec);

// Populate vector with random values ranging between 0 and 1
void fill_vector(vec_t *vec);

// Check that all elements two vectors are nearly equal
bool check_vector(vec_t *x, vec_t *y);

// Print vector
void show_vector(vec_t *vec);

////////////////// SPARSE MATRICES ////////////////////////////

// Allocate all storage for CSR matrix
// Does not set up any indices
csr_t *new_csr(index_t nrow, index_t nnz);

// Free all storage for a matrix
void free_csr(csr_t *m);

// Generate matrix with specified density, and with row/column sizes
// having different distribution characteristics
csr_t *gen_csr(index_t nrow, float density, dist_t dist);

// Print matrix
void show_csr(csr_t *m);

// CSR File format:
// nrow nnz
// rstart[0]
// ...
// rstart[nrow]
// cindex[0] value[0]
// ...
// cindex[nnz-1] value[nnz-1]

// Save CSR Matrix to file
void save_csr(csr_t *m, FILE *outfile);

// Load CSR Matrix from file
csr_t *read_csr(FILE *infile, const char *fname);

// Baseline function for doing sparse-matrix multiply (y = Mx)
void smvp_baseline(csr_t *m, vec_t *x, vec_t *y);


