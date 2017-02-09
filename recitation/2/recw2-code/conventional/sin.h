typedef void (*sin_fun_t)(int N, int terms, float * x, float *result);

void sinx_reference(int N, int terms, float * x, float *result);

typedef struct {
    const char *name;
    sin_fun_t fun;
} benchmark_t;

void init_benchmarks(benchmark_t *b);
