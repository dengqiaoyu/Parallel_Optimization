#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>

#include "CycleTimer.h"

#include "matrix.h"

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -m  --method  <TYPE> Run specified matrix multiplication method\n");
    printf("\tOptions:\n");
    printf("\t\tsimple\tStandard multiplication\n");
    printf("\t\ttranspose\tTranspose B first\n");
    printf("\t\tblock\tUse blocking\n");
    printf("\t\ttblock\tTranspose B first and then use blocking\n");
    printf("\t\tfblock\tTranspose B first and then use optimized blocking\n");
    printf("\t\tcosimple\tCuda, standard multiplication, with less efficient indexing\n");
    printf("\t\tcsimple\tCuda, standard multiplication\n");
    printf("\t\tctranspose\tCuda, transpose B first\n");
    printf("\t\tcoblock\tCuda, Using blocks, but with less efficient indexing\n");
    printf("\t\tcblock\tCuda, Using blocks\n");
    printf("\t\tnvidia\tCuda, Using nVidia code\n");
    printf("\t\tcsblock\tCuda, Using small blocks\n");
    printf("\t\tcqblock\tCuda, Using blocks with 4 elements/thread\n");
    printf("\t\tall\tCycle through all methods\n");
    printf("  -n  --minimum <INT>  Specify minimum value of N\n");
    printf("  -N  --maximum <INT>  Specify maximum value of N\n");
    printf("  -l  --linear        Use linear scaling\n");
    printf("  -h  --help          This message\n");
}

static const char *mcodes[15] = {"simple", "transpose", "block", "tblock", "fblock",
				 "cosimple", "csimple", "ctranspose", "coblock", "cblock", "nvidia", "csblock", "cqblock",
				 "all", "none"};

static const mmul_t mvals[15] = { MMUL_REFERENCE, MMUL_TRANSPOSE, MMUL_BLK, MMUL_TRANSPOSE_BLK, MMUL_FAST_BLK,
				  MMUL_CUDA_OLD_REFERENCE, MMUL_CUDA_REFERENCE, MMUL_CUDA_TRANSPOSE, MMUL_CUDA_OLD_BLK, MMUL_CUDA_BLK, MMUL_CUDA_NVIDIA, MMUL_CUDA_SMALL_BLK, MMUL_CUDA_QUAD_BLK,
				  MMUL_ALL, MMUL_NONE };

static const int cuda_flags[15] = {0, 0, 0, 0, 0,
				   1, 1, 1, 1, 1, 1, 1, 1,
				   0, 0};

static mmul_t find_method(char *name, int *use_cuda_ptr) {
    int i;
    for (i = 0; mvals[i] != MMUL_NONE; i++)
	if (strcmp(name, mcodes[i]) == 0)
	    break;
    if (use_cuda_ptr)
	*use_cuda_ptr = cuda_flags[i];
    return mvals[i];
}

static const char *method_name(mmul_t method) {
    int i;
    for (i = 0; mvals[i] != MMUL_NONE; i++)
	if (mvals[i] == method)
	    break;
    return mcodes[i];
}


static void run(int nMin, int nMax, int use_cuda, int linear, mmul_t method) {
    /* Run test(s) */
    int *NVals  = new int[32];
    double *gfVals = new double[32];
    int N = nMin;
    int tcount = 0;
    for (tcount = 0; tcount < 32 && N <= nMax; tcount++) {
	NVals[tcount] = N;
	gfVals[tcount] = use_cuda ? cudaBenchMM(N, method) : benchMM(N, method);
	if (linear)
	    N += nMin;
	else
	    N += N;
    }
    /* Format report */
    printf("| N");
    for (int t = 0; t < tcount; t++) {
	printf("\t%d", NVals[t]);
    }
    printf("\n");
    printf("| GF");
    for (int t = 0; t < tcount; t++) {
	printf("\t%.2f", gfVals[t]);
    }
    printf("\t%s\n", method_name(method));
}

int main(int argc, char** argv)
{
    int nMin = 8;
    int nMax = 1024;
    mmul_t method = MMUL_REFERENCE;
    int use_cuda = 0;
    int linear = 0;
    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"method",      1, 0, 'm'},
        {"minimum",     1, 0, 'n'},
        {"maximum",     1, 0, 'N'},
        {"linear",     1, 0, 'l'},
        {"help",        0, 0, 'h'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "m:n:N:hl", long_options, NULL)) != EOF) {
        switch (opt) {
        case 'm':
            method = find_method(optarg, &use_cuda);
            break;
        case 'n':
            nMin = atoi(optarg);
            break;
	case 'N':
	    nMax = atoi(optarg);
	    break;
	case 'l':
	    linear = 1;
	    break;
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////
    if (method == MMUL_ALL) {
	for (int i = 0; mvals[i] != MMUL_ALL; i++) {
	    method = mvals[i];
	    use_cuda = cuda_flags[i];
	    run(nMin, nMax, use_cuda, linear, method);
	}
    } else {
	if (use_cuda)
	    printCudaInfo();
	run(nMin, nMax, use_cuda, linear, method);
    }
    return 0;
}
   
