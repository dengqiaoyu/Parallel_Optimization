#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <math.h>


#include "reduce.h"

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -n  --length <INT>  Number of elements in vectors\n");
    printf("  -r  --runs   <INT>  Number of times to repeat test\n");
    printf("  -b  --tpb    <INT>  Number of threads per block\n");
    printf("  -d  --degree <INT>  Reduction degree\n");
    printf("  -h  --help          This message\n");
}

void checkResult(float val, float targetVal) {
    float num = 2.0 * fabs(targetVal - val);
    float denom = fabs(targetVal + val);
    float err = denom == 0 ? num : num/denom;
    if (err > reduceTolerance) {
	printf("ERROR: Reduction should have given %.4f.  Got %.4f instead (Error = %.3f)\n",
	       targetVal, val, err);
    }
}

// Sample implementation
float sumVector(int length, float *data) {
    float val = 0.0;
    for (int i = 0; i < length; i++)
	val += data[i];
    return val;
}

static float randomData(int length, float *data) {
    float val = 0.0;
    for (int i = 0; i < length; i++) {
	// Random numbers between 1.0 and 10.0
	data[i] = 1.0 + ((float) rand()/RAND_MAX) * 9.0;
	//	printf("Value[%d] = %f\n", i, data[i]);
	val += data[i];
    }
    return val;
}

static float fixedData(int length, float *data) {
    float val = 0.0;
    int i;
    float base = 1.0;
    float frac = 0.01;
    for (i = 0; i < 10 && i < length; i++) {
	data[i] = base + frac;
	base *= 10.0;
	val += data[i];
    }
    // Fill remainder with random values
    if (i < length)
	val += randomData(length-i, data+i);
    printf("Input data:");
    for (i = 0; i < 13 && i < length; i++)
	printf("\t%.2f", data[i]);
    if (length > i)
	printf("\t...\tSum = %.2f\n", val);
    else
	printf("\tSum = %.2f\n", val);
    return val;
}

static void run_test(int length) {
    float *data = new float[length];
#if DEBUG
    float targetVal = fixedData(length, data);
#else    
    float targetVal = randomData(length, data);
#endif
    double gflops;
    gflops = cudaReduce(length, data, targetVal);
    printf(
"GFLOPS = %.2f. Length = %d, degree = %d, runs = %d, threads/block = %d\n",
        gflops, length, reduceDegree, reduceRuns, reduceThreadsPerBlock);

}

int main(int argc, char** argv)
{
    int length = reduceLength;
    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"length",     1, 0, 'n'},
        {"degree",     1, 0, 'd'},
        {"runs",       1, 0, 'r'},
        {"tpb",        1, 0, 'b'},
        {"help",       0, 0, 'h'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "n:d:r:b:h", long_options, NULL)) != EOF) {
        switch (opt) {
        case 'n':
            length = atoi(optarg);
            break;
	case 'd':
	    reduceDegree = atoi(optarg);
	    break;
	case 'r':
	    reduceRuns = atoi(optarg);
	    break;
	case 'b':
	    reduceThreadsPerBlock = atoi(optarg);
	    break;
        case 'h':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options 

    //    printCudaInfo();
    run_test(length);
    return 0;
}
