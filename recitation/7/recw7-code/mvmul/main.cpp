// Benchmarking of matrix-vector product programs

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <getopt.h>

#include "matrix.h"
#include "mbench.h"

static void usage(char *pname) {
    printf("Usage: %s [-l LCNT] [-d DPCT] [-t TYP] [-r RUNS]\n", pname);
    printf("    -l LCNT     Set log of matrix size \n");
    printf("    -d DNS      Set matrix percent density\n");
    printf("    -t (u|r|s)  Set matrix type (uniform, random, skewed)\n");
    printf("    -r RUNS     Set number of runs to time over\n");
    exit(0);
}


int main(int argc, char *argv[]) {
    int pscale = 3;
    int dpct = 10;
    int nruns = 100;
    dist_t dist = DIST_UNIFORM;
    int c;
    int dc;

    while ((c = getopt(argc, argv, "l:d:t:r:")) != -1) {
	switch(c) {
	case 'l':
	    pscale = atoi(optarg);
	    break;
	case 'd':
	    dpct = atoi(optarg);
	    break;
	case 't':
	    dc = optarg[0];
	    dist = find_dist(dc);
	    if (dist == DIST_BAD) {
		printf("Unknown distribution type '%c'\n", dc);
		exit(1);
	    }
	    break;
	case 'r':
	    nruns = atoi(optarg);
	    break;
	default:
	    usage(argv[0]);
	}
    }

    register_functions();
    run_all(pscale, dpct, dist, nruns);
    return 0;
}

