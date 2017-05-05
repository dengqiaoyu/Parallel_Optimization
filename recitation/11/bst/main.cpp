#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include "tsynch.h"
#include "bst.h"
#include "driver.h"

void usage(char *name) {
    printf("Usage: %s <OPTIONS>\n", name);
    printf("   -i          Only run insertions concurrently\n");
    printf("   -v VERB     Set verbosity level (0-3)\n");
    printf("   -m (n|g|s|h)  Run with no/global/simple/h-o-h locking\n");
    printf("   -t THREADS  Set number of threads\n");
    printf("   -p PHASES   Set maximum phases to run\n");
    printf("   -s SIZE     Set maximum set size\n");
    printf("   -n OPS      Set number of operations\n");
    printf("   -P PROB     Set probability of performing a sum operation\n");
    exit(0);
}

const char *show_mode(synch_t mode) {
    if (mode == NO_SYNCH)
	return "No synchronization";
    if (mode == SIMPLE_SYNCH)
	return "Fine-grained, but not hand-over-hand synchronization";
    if (mode == FULL_SYNCH)
	return "Hand-over-hand synchronization";
    if (mode == GLOBAL_SYNCH)
	return "Global synchronization";
    return "Unknown synchronization";
}

int main(int argc, char *argv[]) {
    int nthreads = 1;
    int size = 20;
    int ops = 100;
    int phases = 1;
    int vlevel = 0;
    int c;
    synch_t mode = NO_SYNCH;
    char mchar;
    bool insertOnly = false;
    double sprob = 0.0;
    while ((c = getopt(argc, argv, "hiv:m:t:p:s:n:P:")) != -1) {
	switch(c) {
	case 'i':
	    insertOnly = true;
	    break;
	case 'm':
	    mchar = optarg[0];
	    switch(mchar) {
	    case 'n':
		mode = NO_SYNCH;
		break;
	    case 'g':
		mode = GLOBAL_SYNCH;
		break;
	    case 's':
		mode = SIMPLE_SYNCH;
		break;
	    case 'h':
		mode = FULL_SYNCH;
		break;
	    default:
		printf("Unknown mode '%c'\n", mchar);
		usage(argv[0]);
		break;
	    }
	    break;
	case 't':
	    nthreads = atoi(optarg);
	    break;
	case 'p':
	    phases = atoi(optarg);
	    break;
	case 's':
	    size = atoi(optarg);
	    break;
	case 'n':
	    ops = atoi(optarg);
	    break;
	case 'v':
	    vlevel = atoi(optarg);
	    break;
	case 'P':
	    sprob = atof(optarg);
	    if (sprob > 1.0)
		// Assume it's a percentage
		sprob /= 100.0;
	    break;
	default:
	    printf("Unknown option '%c'\n", c);
	case 'h':
	    usage(argv[0]);
	    
	}
    }
    set_verb_level(vlevel);
    report(1, "Creating tester with %d threads", nthreads);
    report(1, "Running in mode: %s", show_mode(mode));
    Tester *t = new Tester(nthreads);
    int p;
    for (p = 0; p < phases; p++) {
	report(1, "Starting phase %d.  Size = %d, Ops = %d", p, size, ops);
	init_driver(mode, size, sprob, insertOnly);
	t->runPhase(drive, ops);
	free_driver();
    }
    report(3, "Shutting down tester");
    t->shutdown();
    report(1, "Testing completed");

    return 0;
}
