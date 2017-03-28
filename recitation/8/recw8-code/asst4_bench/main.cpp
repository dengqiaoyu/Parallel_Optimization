#include <iostream>
#include <getopt.h>
#include <stdlib.h>

#include "mylogging.h"
#include "benchmark.h"
#include "stat.h"
#include "run.h"

// Parse colon-separated list of integers
void split_params(char *s, int *param, int target) {
    char *p = s;
    while (target > 0 && p && *p) {
	long val = strtol(p, &p, 10);
	*param = val;
	param++;
	target--;
	if (p && *p == ':')
	    p++;
    }
    if (p && *p) {
	std::cerr << "Invalid parameter list '" << s << "'" << std::endl;
	std::cerr << "Unexpected stuff '" << p << "'" << std::endl;
	exit(1);
    }
    if (target != 0) {
	std::cerr << "Invalid parameter list '" << s << "'" << std::endl;
	std::cerr << target << " missing values" << std::endl;
	exit(1);
    }
}

void usage(char *name) {
    std::cout << "Usage: " << name << " -t THREADS -j JOBLIST -l LIMITLIST" << std::endl;
    std::cout << "   -t THREADS     Set number of threads" << std::endl;
    std::cout << "   -j JOBLIST     Set number of jobs of each type (colon-separated list)" << std::endl;
    std::cout << "   -l LIMITS      Set limits for each type (colon-separated list)" << std::endl;
    exit(0);
}

int main(int argc, char *argv[]) {
    int c;
    int nthreads = 32;
    while ((c = getopt(argc, argv, "ht:j:l:")) != -1) {
	switch (c) {
	case 'h':
	    usage(argv[0]);
	    break;
	case 't':
	    nthreads = atoi(optarg);
	    break;
	case 'j':
	    split_params(optarg, job_count, BENCH_END);
	    break;
	case 'l':
	    split_params(optarg, job_limit, BENCH_END);
	    break;
	default:
	    usage(argv[0]);
	    break;
	}
    }
    init_logging(INFO, argv[0]);
    run(nthreads);
    finish_logging();
    return 0;
}
