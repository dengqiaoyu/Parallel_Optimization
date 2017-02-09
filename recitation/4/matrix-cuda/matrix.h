

// Different ways of performing matrix multiplication
enum mmul_t {
    MMUL_REFERENCE,        // Generic matrix multiplication
    MMUL_TRANSPOSE,        // Transpose B before multiplying
    MMUL_BLK,              // Use blocks
    MMUL_TRANSPOSE_BLK,    // Transpose B and use blocks
    MMUL_FAST_BLK,       // Transpose B and use optimized blocks
    MMUL_VECTOR,           // Transpose B and use vector instructions
    MMUL_CUDA_OLD_REFERENCE,   // Standard CUDA implementation, less efficient indexing
    MMUL_CUDA_REFERENCE,   // Standard CUDA implementation
    MMUL_CUDA_TRANSPOSE,   // CUDA.  Transpose B before multiplying
    MMUL_CUDA_OLD_BLK,       // CUDA.  Use blocks.  Inefficient indexing
    MMUL_CUDA_BLK,       // CUDA.  Use blocks
    MMUL_CUDA_BLK32,       // CUDA.  N must be multiple of 32
    MMUL_CUDA_NVIDIA,       // CUDA.  Using nVidia code
    MMUL_CUDA_SMALL_BLK, // CUDA.  Use small blocks
    MMUL_CUDA_QUAD_BLK, // CUDA.  Use blocks with 4 elements / block
    MMUL_ALL,              // Cycle through entire repetoire
    MMUL_NONE             // Do nothing
};

// Tunable parameters
#ifndef MATRIX
// Relative error tolerance
extern float errTolerance;
// Limit of error reporting
extern int errLimit;
// Optimum number of FP operations for measurement
extern long targetOps;
#endif /* MATRIX */

/* Useful macros */
/* Find element based on row-major ordering */
#define RM(r, c, width) ((r) * (width) + (c))
/* Find element based on row-major ordering for 3 dimensions */
#define RM3(r, c, d, width) RM(RM(r, c, width), d, width)

/* Find element based on column-major ordering */
#define CM(r, c, height) ((c) * (height) + (r))

/* Blocking parameters */
/* Small block size */
#define SBLK 8
/* Medium block size */
#define MBLK 16
/* Large block size */
#define LBLK 32





// From matrix.cpp

// Benchmark function returns the gigaflops for the operations,
// including any transpose cost
double benchMM(int N, mmul_t method);
// Test two matrices for equality.  Return number of mismatches
int checkMatrix(int N, float *matTest, float *matGood);
void setup(int N);

// From cudaMatrix.cu

void cudaSetup(int N, float *aData, float *bData, float *gData);
double cudaBenchMM(int N, mmul_t method);
void printCudaInfo();

