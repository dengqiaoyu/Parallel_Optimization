/* Radix sorting */

typedef unsigned long data_t;
typedef unsigned index_t;

// How many bits does each pass use
#define BASE_BITS 8
// What is radix of sort
#define BASE (1 << BASE_BITS)
// Mask of all 1's over BASE_BITS
#define MASK (BASE-1)
// Extract sorting key from data
#define DIGITS(v, shift) (((v) >> (shift)) & MASK)

// Type of function that sorts N values, copying from
// copying from indata to outdata and while using scratchdata
typedef void (*sort_fun_t)(index_t N, data_t *indata,
		      data_t *outdata, data_t *scratchdata);

// Functions defined in rsort.cpp

// Sorter implemented with qsort library function
void lib_sort(index_t N, data_t *indata,
		      data_t *outdata, data_t *scratchdata);

// Register all sorting functions
void register_all();

// Functions defined in sbench.cpp

void register_sorter(sort_fun_t sfun, const char *name);

void test_sorters(index_t N, index_t reps, bool verbose);




