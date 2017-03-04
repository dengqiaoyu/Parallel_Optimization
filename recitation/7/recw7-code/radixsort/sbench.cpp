/* Run benchmark of sorters */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "CycleTimer.h"

#include "rsort.h"

data_t *indata;
data_t *outdata;
data_t *scratchdata;
data_t *compdata;

#define MAXSORTERS 25

static struct {
    sort_fun_t sfun;
    const char *name;
} sorters[MAXSORTERS];

index_t sorter_count = 0;


static data_t random_data() {
    index_t bcount = 16;
    data_t mask = (1<<bcount)-1;
    data_t v = -1;
    for (index_t i = 0; i < sizeof(data_t)*8; i+=bcount)
	v = (v << bcount) | (random() & mask);
    return v;
}

static void init(index_t N, bool verbose) {
    indata = (data_t *) calloc(N, sizeof(data_t));
    outdata = (data_t *) calloc(N, sizeof(data_t));
    scratchdata = (data_t *) calloc(N, sizeof(data_t));
    compdata = (data_t *) calloc(N, sizeof(data_t));

    if (verbose) {
	printf("Input data:\n");
	for (index_t i = 0; i < N; i++) {
	    indata[i] = random_data();
	    printf("\t%lu\t%16lx\n",
		   (unsigned long) i, (unsigned long) indata[i]);
	}
    }
    lib_sort(N, indata, compdata, scratchdata);
    if (verbose) {
	printf("Correctly sorted data:\n");
	for (index_t i = 0; i < N; i++) {
	    printf("\t%lu\t%16lx\n",
		   (unsigned long) i, (unsigned long) compdata[i]);
	}
    }
}

void register_sorter(sort_fun_t sfun, const char *name) {
    sorters[sorter_count].sfun = sfun;
    sorters[sorter_count].name = name;
    sorter_count++;
}

bool check_sorter(index_t N, sort_fun_t sfun, const char *name, bool verbose) {
    memset(outdata, 0, N * sizeof(data_t));
    sfun(N, indata, outdata, scratchdata);
    printf("Sorting with %s\n", name);
    if (verbose) {
	for (index_t i = 0; i < N; i++) {
	    printf("\t%lu\t%16lx\n", (unsigned long) i,
		   (unsigned long) outdata[i]);
	}
    }
    for (index_t i = 0; i < N; i++) {
	if (outdata[i] != compdata[i]) {
	    printf("Sort function %s.  Index %lu.  Expected %lu.  Got %lu\n",
		   name, (unsigned long) i, (unsigned long) compdata[i],
		   (unsigned long) outdata[i]);
	    return false;
	}
    }
    return true;
}



void test_sorters(index_t N, index_t reps, bool verbose) {
    init(N, verbose);
    for (index_t s = 0; s < sorter_count; s++) {
	sort_fun_t sfun = sorters[s].sfun;
	const char *name = sorters[s].name;
	if (!check_sorter(N, sfun, name, verbose))
	    continue;
	double startTime = CycleTimer::currentSeconds();
	for (index_t r = 0; r < reps; r++)
	    sfun(N, indata, outdata, scratchdata);
	double endTime = CycleTimer::currentSeconds();
	double rate = 1e-6 * (double) N * reps / (endTime-startTime);
	printf("N:\t%lu\tRate:\t\%.3f\tFun:\t%s\n",
	       (unsigned long) N, rate, name);
    }
}
