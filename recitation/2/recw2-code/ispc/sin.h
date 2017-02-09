typedef void (*sin_fun_t)(int32_t N, int32_t terms, float * x, float *result);

// void sinx_reference(int32_t N, int32_t terms, float * x, float *result);

typedef struct {
    const char *name;
    sin_fun_t fun;
} benchmark_t;

