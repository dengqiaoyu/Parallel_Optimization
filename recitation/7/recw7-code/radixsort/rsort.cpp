/* Code to sort array of long unsigned via radix sort */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "rsort.h"

// Comparison function for library quicksort
int dcomp(const void *vp1, const void *vp2) {
    data_t v1 = *(data_t *) vp1;
    data_t v2 = *(data_t *) vp2;
    if (v1 < v2)
	return -1;
    if (v1 > v2)
	return 1;
    return 0;
}

// Using library quicksort
void lib_sort(index_t N, data_t *indata,
	      data_t *outdata, data_t *scratchdata) {
    memcpy(outdata, indata, N * sizeof(data_t));
    qsort((void *) outdata, N, sizeof(data_t), dcomp);
}

// Radix sorting, sequential.
void seq_radix_sort(index_t N, data_t *indata,
	      data_t *outdata, data_t *scratchdata) {
    data_t *src = indata;
    // Assume even number of steps, so by toggling between
    // outdata and scratchdata, result will end up in outdata
    data_t *dest = scratchdata;
    index_t count[BASE];
    index_t offset[BASE];
    index_t total_digits = sizeof(data_t) * 8;

    for (index_t shift = 0; shift < total_digits; shift+= BASE_BITS) {
	memset(count, 0, BASE*sizeof(index_t));
	// Accumulate count for each key value
	for (index_t i = 0; i < N; i++) {
	    data_t key = DIGITS(src[i], shift);
	    count[key]++;
	}
	// Compute offsets
	offset[0] = 0;
	for (index_t b = 1; b < BASE; b++)
	    offset[b] = offset[b-1] + count[b-1];
	// Distribute data
	for (index_t i = 0; i < N; i++) {
	    data_t key = DIGITS(src[i], shift);
	    index_t pos = offset[key]++;
	    dest[pos] = src[i];
	}
	// Find new src & dest
	src = dest;
	dest = (dest == outdata) ? scratchdata : outdata;
    }
}

// Radix sorting, parallel.  Use atomic updates + critical sections
void critical_par_radix_sort(index_t N, data_t *indata,
	      data_t *outdata, data_t *scratchdata) {
    data_t *src = indata;
    // Assume even number of steps, so by toggling between
    // outdata and scratchdata, result will end up in outdata
    data_t *dest = scratchdata;
    index_t count[BASE];
    index_t offset[BASE];
    index_t total_digits = sizeof(data_t) * 8;

    for (index_t shift = 0; shift < total_digits; shift+= BASE_BITS) {
	memset(count, 0, BASE*sizeof(index_t));
	// Accumulate count for each key value
        #pragma omp parallel for schedule(static)
	for (index_t i = 0; i < N; i++) {
	    data_t key = DIGITS(src[i], shift);
            #pragma omp atomic
	    count[key]++;
	}
	// Compute offsets
	offset[0] = 0;
	for (index_t b = 1; b < BASE; b++)
	    offset[b] = offset[b-1] + count[b-1];
	// Distribute data
        #pragma omp parallel for schedule(static)
	for (index_t i = 0; i < N; i++) {
	    data_t key = DIGITS(src[i], shift);
	    index_t pos;
            #pragma omp critical
	    pos = offset[key]++;
	    dest[pos] = src[i];
	}
	// Find new src & dest
	src = dest;
	dest = (dest == outdata) ? scratchdata : outdata;
    }
}

#if 0
// Does not compile on with older versions of OpenMP
// Avoid critical section by using atomic capture
void fetchadd_par_radix_sort(index_t N, data_t *indata,
	      data_t *outdata, data_t *scratchdata) {
    data_t *src = indata;
    // Assume even number of steps, so by toggling between
    // outdata and scratchdata, result will end up in outdata
    data_t *dest = scratchdata;
    index_t count[BASE];
    index_t offset[BASE];
    index_t total_digits = sizeof(data_t) * 8;

    for (index_t shift = 0; shift < total_digits; shift+= BASE_BITS) {
	memset(count, 0, BASE*sizeof(index_t));
	// Accumulate count for each key value
        #pragma omp parallel for schedule(static)
	for (index_t i = 0; i < N; i++) {
	    data_t key = DIGITS(src[i], shift);
            #pragma omp atomic
	    count[key]++;
	}
	// Compute offsets
	offset[0] = 0;
	for (index_t b = 1; b < BASE; b++)
	    offset[b] = offset[b-1] + count[b-1];
	// Distribute data
        #pragma omp parallel for schedule(static)
	for (index_t i = 0; i < N; i++) {
	    data_t key = DIGITS(src[i], shift);
	    index_t pos;
	    #pragma omp atomic capture
	    pos = offset[key]++;
	    dest[pos] = src[i];
	}
	// Find new src & dest
	src = dest;
	dest = (dest == outdata) ? scratchdata : outdata;
    }
}
#endif

// Modified version of code due to Haichuan Wang, UIUC
void uiuc_radix_sort(index_t N, data_t *indata,
		     data_t *outdata, data_t *scratchdata) {
    data_t *src = indata;
    // Assume even number of steps, so by toggling between
    // outdata and scratchdata, result will end up in outdata
    data_t *dest = scratchdata;
    index_t total_digits = sizeof(data_t) * 8;
 
    //Each thread use local_bucket to move data
    for(index_t shift = 0; shift < total_digits; shift+=BASE_BITS) {
        index_t bucket[BASE] = {0};
        index_t local_bucket[BASE] = {0}; // size needed in each bucket/thread
        #pragma omp parallel firstprivate(local_bucket)
        {
	    // Each thread generates counts for its data
            #pragma omp for schedule(static) nowait
            for(index_t i = 0; i < N; i++){
		data_t key = DIGITS(src[i], shift);
                local_bucket[key]++;
            }
	    // Combine counts for all threads
            #pragma omp critical
            for(index_t b = 0; b < BASE; b++) {
                bucket[b] += local_bucket[b];
            }
            #pragma omp barrier
	    // Convert counts into offsets
            #pragma omp single
            for (index_t b = 1; b < BASE; b++) {
                bucket[b] += bucket[b-1];
            }
            int nthreads = omp_get_num_threads();
            int tid = omp_get_thread_num();
	    // Generate local offsets for each thread
	    // Each thread will be activated once.
            for(int t = nthreads-1; t >= 0; t--) {
                if(t == tid) {
                    for(index_t b = 0; b < BASE; b++) {
                        bucket[b] -= local_bucket[b];
                        local_bucket[b] = bucket[b];
                    }
                } else { //just do barrier
                    #pragma omp barrier
                }
 
            }
	    // Distribute data using local offsets
            #pragma omp for schedule(static)
            for(index_t i = 0; i < N; i++) {
		data_t key = DIGITS(src[i], shift);
		index_t pos = local_bucket[key]++;
                dest[pos] = src[i];
            }
        }
	// Find new src & dest
	src = dest;
	dest = (dest == outdata) ? scratchdata : outdata;
    }
}


// Variation of code due to Haichuan Wang, UIUC
void full_par_radix_sort(index_t N, data_t *indata,
		     data_t *outdata, data_t *scratchdata) {
    data_t *src = indata;
    // Assume even number of steps, so by toggling between
    // outdata and scratchdata, result will end up in outdata
    data_t *dest = scratchdata;
    index_t total_digits = sizeof(data_t) * 8;
    index_t count[BASE];
    index_t offset[BASE];
 
    for(index_t shift = 0; shift < total_digits; shift+=BASE_BITS) {
	memset(count, 0, BASE*sizeof(index_t));
        #pragma omp parallel
        {
	    // Per-thread counts and offsets
	    index_t local_count[BASE] = {0};
	    index_t local_offset[BASE];
            #pragma omp for schedule(static) nowait
	    // Gather counts on per-thread basis
            for(index_t i = 0; i < N; i++){
		data_t key = DIGITS(src[i], shift);
                local_count[key]++;
            }
	    // Compute global counts based on local ones
	    // Critical faster than each addition being atomic
            #pragma omp critical  
            for(index_t b = 0; b < BASE; b++) {
                count[b] += local_count[b];
            }
	    #pragma omp barrier
	    // Compute global offsets
            #pragma omp single
	    {
		offset[0] = 0;
		for (index_t b = 1; b < BASE; b++)
		    offset[b] = count[b-1] + offset[b-1];
	    }
            int nthreads = omp_get_num_threads();
            int tid = omp_get_thread_num();
	    // Compute local offsets
	    // Enforce serialization by triggering each thread in sequence
	    for (int t = 0; t < nthreads; t++) {
                if(t == tid) {
                    for(index_t b = 0; b < BASE; b++) {
                        local_offset[b] = offset[b];
                        offset[b] += local_count[b];
                    }
                }
                #pragma omp barrier 
            }
            #pragma omp for schedule(static)
            for(index_t i = 0; i < N; i++) {
		data_t key = DIGITS(src[i], shift);
		index_t pos = local_offset[key]++;
                dest[pos] = src[i];
            }
        }
	// Find new src & dest
	src = dest;
	dest = (dest == outdata) ? scratchdata : outdata;
    }
}



void register_all() {
    register_sorter(lib_sort, "Library quicksort");
    register_sorter(seq_radix_sort, "Radix sort: sequential");
    //    register_sorter(critical_par_radix_sort, "Radix sort: simple parallel with critical section");
    //    register_sorter(fetchadd_par_radix_sort, "Radix sort: simple parallel using fetch and add");
    register_sorter(uiuc_radix_sort, "Radix sort: UIUC version");
    register_sorter(full_par_radix_sort, "Radix sort: Modified UIUC version");
}
