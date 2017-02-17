// Benchmarking of matrix-vector products

//// In file mbench.cpp
// Allow registration of multiple functions
void register_rvp_csr(const char *name, rvp_csr_t rp_fun);
void register_mvp_csr(const char *name, mvp_csr_t mp_fun);

void register_rvp_dense(const char *name, rvp_dense_t rp_fun);
void register_mvp_dense(const char *name, mvp_dense_t mp_fun);

// Run benchmark.  Return result in gigaflops
// Parameters:
// pscale = log_10(nrow)
// dpct = density percent
// dist = distribution type
float run_bench(int pscale, int dpct, dist_t dist, bool densify,
	        const char *rvp_name, const char *mvp_name);

void run_all_sparse(int pscale, int dpct, dist_t dist);

void run_all_dense(int pscale, int dpct, dist_t dist);

//// In file mvmul.cpp
void register_functions();

