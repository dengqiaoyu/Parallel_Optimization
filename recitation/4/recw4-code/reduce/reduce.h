// Inject different errors

#ifndef DEBUG
#define DEBUG 0
#endif

#if DEBUG
#define THREADSPERBLOCK 32
#else
#define THREADSPERBLOCK 1024
#endif

// Controlling parameters.  Defined in reduce.cu
#ifndef REDUCE
extern int reduceDegree;
extern int reduceThreadsPerBlock;
extern int reduceRuns;
extern float reduceTolerance;
extern int reduceLength;
#endif

// Defined in main.cpp
void checkResult(float val, float targetVal);

// Defined in reduce.cu

// All of these functions return the gigaflops for the operations,
//   not counting data transfer
double cudaReduce(int length, float *avec, float targetVal);

void printCudaInfo();
