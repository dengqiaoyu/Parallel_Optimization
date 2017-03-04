// Benchmarking of matrix-vector products

//// In file mbench.cpp
// Allow registration of multiple functions
void register_mult_csr(const char *name, mult_csr_t mult_fun);


// Run benchmark.  Return result in gigaflops
// Parameters:
// pscale = log_10(nrow)
// dpct = density percent
// dist = distribution type
float run_bench(int pscale, int dpct, dist_t dist,
		const char *mult_name);

void run_all(int pscale, int dpct, dist_t dist, int nruns);

//// In file mvmul.cpp
void register_functions();

