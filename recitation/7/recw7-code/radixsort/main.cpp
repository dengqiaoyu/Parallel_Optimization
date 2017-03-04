#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <stdbool.h>

#include "rsort.h"


static void usage(char *pname) {
    printf("Usage: %s [-n N] [-r R] [-v]\n", pname);
    printf("    -n N Set number of elements to sort\n");
    printf("    -r R Set number repetitions\n");
    printf("    -v   Run in verbose mode\n");
    exit(0);
}

int main(int argc, char *argv[]) {
    index_t N = 100;
    index_t reps = 20;
    bool verbose = false;
    int c;
    while ((c = getopt(argc, argv, "n:r:v")) != -1) {
	switch(c) {
	case 'n':
	    N = atoi(optarg);
	    break;
	case 'r':
	    reps = atoi(optarg);
	    break;
	case 'v':
	    verbose = true;
	    break;
	default:
	    usage(argv[0]);
	}
    }
    register_all();
    test_sorters(N, reps, verbose);
    return 0;
}
