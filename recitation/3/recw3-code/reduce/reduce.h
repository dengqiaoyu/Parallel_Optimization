#define THREADSPERBLOCK 1024

// Controlling parameters
#ifndef REDUCE
extern int reduceDegree;
extern int reduceThreadsPerBlock;
extern int reduceRuns;
extern float reduceTolerance;
#endif

// Defined in main.cpp
void checkResult(float val, float targetVal);

// Defined in reduce.cu

// All of these functions return the gigaflops for the operations, not counting data transfer
double cudaReduce(int length, float *avec, float targetVal);

void printCudaInfo();
