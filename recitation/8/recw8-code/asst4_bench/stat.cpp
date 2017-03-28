// Collect and report run information
#include <iostream>
#include <algorithm>

#include "benchmark.h"
#include "stat.h"
#include "run.h"

// Configuration
const char *job_names[BENCH_END] = { "wisdom", "primes", "tellme", "project" };

// Statistics gathering (all in ms)
int job_min[BENCH_END] = { 0 };
int job_max[BENCH_END] = { 0 };
int job_total[BENCH_END] = { 0 };

void report_job(int benchmark, int msecs) {
    job_total[benchmark] += msecs;
    if (job_min[benchmark] == 0)
	job_min[benchmark] = msecs;
    else
	job_min[benchmark] = std::min(job_min[benchmark], msecs);
    job_max[benchmark] = std::max(job_max[benchmark], msecs);
}

void report(int nthreads, double secs) {
    std::cout << nthreads << " threads" << std::endl;
    for (int b = BENCH_WISDOM; b < BENCH_END; b++) {
	std::cout << "\t" << job_names[b];
    }
    std::cout << std::endl;

    std::cout << "Jobs:";
    for (int b = BENCH_WISDOM; b < BENCH_END; b++) {
	std::cout << "\t" << job_count[b];
    }
    std::cout << std::endl;

    std::cout << "Limits:";
    for (int b = BENCH_WISDOM; b < BENCH_END; b++) {
	std::cout << "\t" << job_limit[b];
    }
    std::cout << std::endl;

    std::cout << "Min:";
    for (int b = BENCH_WISDOM; b < BENCH_END; b++) {
	if (job_count[b] > 0)
	    std::cout << "\t" << job_min[b];
	else
	    std::cout << "\t--";
    }
    std::cout << std::endl;

    std::cout << "avg:";
    for (int b = BENCH_WISDOM; b < BENCH_END; b++) {
	int avg = job_count[b] ? job_total[b]/job_count[b] : 0;
	if (job_count[b] > 0)
	    std::cout << "\t" << avg;
	else
	    std::cout << "\t--";
    }
    std::cout << std::endl;

    std::cout << "Max:";
    for (int b = BENCH_WISDOM; b < BENCH_END; b++) {
	if (job_count[b] > 0)
	    std::cout << "\t" << job_max[b];
	else
	    std::cout << "\t--";
    }
    std::cout << std::endl;

    int msecs = (int) (1000.0 * secs);
    std::cout << "Total time: " << msecs << "ms" << std::endl;
}
