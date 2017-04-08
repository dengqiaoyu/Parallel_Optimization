/**
 * @file scan.cu
 * @brief This file contain three different level improvements on exclusive scan
 *        , and they are naive way, using threads to do useful works and using
 *        shared memory and bank conflict free to further improve performance.
 * @author Qiaoyu Deng(qdeng), Changkai Zhou(zchangka)
 * @bug No known bugs
 */

/* c lib inludes */
#include <stdio.h>

/* cuda lib incljudes */
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

/* thrust lib includs */
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

/* timer include */
#include "CycleTimer.h" /* CycleTimer::currentSeconds(); */

/* Constant Define */
#define THREADS_PER_BLOCK 128
#define THREADS_MAX_PER_BLOCK 1024
#define THREADS_PER_BLOCK_SUM THREADS_MAX_PER_BLOCK
#define THREADS_PER_BLOCK_REPEAT THREADS_MAX_PER_BLOCK
#define MAX_DATA_LENGTH_PER_BLOCK (THREADS_PER_BLOCK * 2)

/* Used by exclusive_scan_sharedmem to reduce bank conflict */
#define CONFLICT_FREE
#ifdef CONFLICT_FREE
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) 0
#endif

extern float toBW(int bytes, float sec);

/* Debug macros. */
// #define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#else
#define cudaCheckError(ans) ans
#endif



/**
 * Return the next value of n that is power of 2
 * @param  n any integer number
 * @return   number that is power of 2
 */
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

/**
 * Print intger array that is located in device memory
 * @param device_memory the pointer to the device memory
 * @param len           the length of that device memory
 */
void print_int_device_memory(int* device_memory, int len) {
    int* device_memmory_debug = new int[len];
    cudaCheckError(cudaMemcpy(device_memmory_debug, device_memory,
                              len * sizeof(int), cudaMemcpyDeviceToHost));
    printf("printing array, length: %d\n", len);
    for (int i = 0; i < len; i++) {
        printf("%d ", device_memmory_debug[i]);
    }
    printf("\n");
    delete[] device_memmory_debug;
}

/**
 * Kernel cundtion that is called by exclusive_scan_sharedmem()
 * @param device_result the input array and output array
 * @param rd_len        the rounded length of array that is greater than
 *                      MAX_DATA_LENGTH_PER_BLOCK
 * @param block_sum     the return array of each block's last thread's value,
 *                      which is the sum of every block
 */
__global__ void es_shm_pcom_perb_kernel(int* device_result,
                                        unsigned int rd_len,
                                        int* block_sum) {
    __shared__ int work_set[MAX_DATA_LENGTH_PER_BLOCK + 64];
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockDim.x * bid + tid;
    unsigned int src1_in_sharemem = 2 * tid + 1;
    unsigned int src2_in_sharemem = 2 * tid;
    unsigned int src1_in_device = 2 * gid + 1;
    unsigned int src2_in_device = 2 * gid;
    /* Set conflict free memory indexs */
    unsigned int src1_offset = CONFLICT_FREE_OFFSET(src1_in_sharemem);
    unsigned int src2_offset = CONFLICT_FREE_OFFSET(src2_in_sharemem);

    // Load input from global memory to shared memory
    if (src1_in_device < rd_len) {
        work_set[src2_in_sharemem + src2_offset] =
            device_result[src2_in_device];
        work_set[src1_in_sharemem + src1_offset] = device_result[src1_in_device]
                + work_set[src2_in_sharemem + src2_offset];
    }
    /**
     * Ensure all of input have been load to shared memory befor any computation
     * begin.
     */
    __syncthreads();

    /**
     * Add items like a tree, and adjust the number of threads that is doing
     * computation at each iteration.
     */
    unsigned int offset = 2;
    for (unsigned int work_thrds = MAX_DATA_LENGTH_PER_BLOCK / 4;
            work_thrds >= 1;
            work_thrds /= 2) {
        if (tid < work_thrds) {
            int src1_index = 2 * offset * (tid + 1) - 1;
            int src1_offseted_index = src1_index
                                      + CONFLICT_FREE_OFFSET(src1_index);
            int src0_index = src1_index - offset;
            int src0_offseted_index = src0_index
                                      + CONFLICT_FREE_OFFSET(src0_index);
            work_set[src1_offseted_index] += work_set[src0_offseted_index];
        }
        /**
         * Ensure each iteration has completed because the iteration will need
         * the last iteration's result.
         */
        __syncthreads();
        offset *= 2;
    }

    /**
     * Let 1 thread to load the last value of the shared memory which is the sum
     * of the current shared array.
     */
    if (tid == blockDim.x - 1) {
        block_sum[bid] = work_set[2 * (tid + 1) - 1];
        work_set[2 * (tid + 1) - 1] = 0;
    }
    __syncthreads();

    /* Do the down sweep process */
    unsigned int work_thrds_limit = MAX_DATA_LENGTH_PER_BLOCK;
    offset = MAX_DATA_LENGTH_PER_BLOCK / 2;
    for (unsigned int work_thrds = 1;
            work_thrds < work_thrds_limit;
            work_thrds *= 2) {
        if (tid < work_thrds) {
            int src1_index = 2 * offset * (tid + 1) - 1;
            int src1_offseted_index =
                src1_index + CONFLICT_FREE_OFFSET(src1_index);
            int src0_index = src1_index - offset;
            int src0_offseted_index =
                src0_index + CONFLICT_FREE_OFFSET(src0_index);
            int swap = work_set[src0_offseted_index];
            work_set[src0_offseted_index] = work_set[src1_offseted_index];
            work_set[src1_offseted_index] += swap;
        }
        __syncthreads();
        offset /= 2;
    }

    /* Load back to global memory */
    if (src1_in_device < rd_len) {
        device_result[src1_in_device] = work_set[src1_in_sharemem + src1_offset] ;
        device_result[src2_in_device] = work_set[src2_in_sharemem + src2_offset] ;
    }
}

/**
 * Add item to the entire arral of after sum array itself has been scan.
 * @param device_result the input array and output array
 * @param block_sum     If the sum array itself has a length greater than 1024,
 *                      then we need to add block sum recusively
 * @param real_num      the actual length of input array
 */
__global__ void add_sum_kernel(int* device_result, int* block_sum,
                               int real_num) {
    unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < real_num) {
        unsigned int block_sum_index = gid / (MAX_DATA_LENGTH_PER_BLOCK);
        device_result[gid] += block_sum[block_sum_index];
    }
}

/**
 * This exclusive scan use shared memory and bank conflict free indexs to
 * improve performance.
 * @param device_result the input array and output array
 * @param rd_len        the rounded length of array that is greater than
 *                      MAX_DATA_LENGTH_PER_BLOCK
 * @param real_len      the actual length of input array
 */
void exclusive_scan_sharedmem(int *device_result, int rd_len, int real_len) {
    int threadsPerBlock = THREADS_PER_BLOCK;
    /* Ensure there is at least one block work on grid */
    int blockPerGrid = std::max(1, rd_len / (MAX_DATA_LENGTH_PER_BLOCK));
    int* block_sum;
    unsigned int rd_block_sum_len = std::max(nextPow2(blockPerGrid),
                                    MAX_DATA_LENGTH_PER_BLOCK);
    cudaMalloc((void **)&block_sum, sizeof(int) * rd_block_sum_len);
    cudaMemset(block_sum, 0, sizeof(int) * rd_block_sum_len);

    // This is intended to go over than 80 lines for better readness
    es_shm_pcom_perb_kernel <<< blockPerGrid, threadsPerBlock>>>(device_result, rd_len, block_sum);
    cudaCheckError(cudaThreadSynchronize());

    /**
     * If the array of sum itself need to be computed in different block, we
     * we need to compute its scan result recursively.
     */
    if (blockPerGrid <= MAX_DATA_LENGTH_PER_BLOCK) {
        int block_sum_blockPerGrid =
            std::max((unsigned int)1,
                     rd_block_sum_len / (MAX_DATA_LENGTH_PER_BLOCK));
        int *block_sum_last = 0;
        cudaMalloc((void **)&block_sum_last, sizeof(int));
        // This is intended to go over than 80 characters for better readness
        es_shm_pcom_perb_kernel <<< block_sum_blockPerGrid, threadsPerBlock>>>(block_sum, rd_block_sum_len, block_sum_last);
        cudaCheckError(cudaThreadSynchronize());
    } else {
        exclusive_scan_sharedmem(block_sum, rd_block_sum_len, blockPerGrid);
        cudaCheckError(cudaThreadSynchronize());
    }

    /* Add sum result to the array */
    threadsPerBlock = THREADS_PER_BLOCK_SUM;
    blockPerGrid = (real_len + threadsPerBlock - 1) / threadsPerBlock;
    // This is intended to go over than 80 lines for better readness
    add_sum_kernel <<< blockPerGrid, threadsPerBlock>>>(device_result, block_sum, real_len);
    cudaCheckError(cudaThreadSynchronize());
    cudaFree(block_sum);
}

/**
 * Up sweep process of naive scan algorithm by adjustting working threads
 * @param device_result  the input array and output array
 * @param twod           offset to find the next source index
 * @param twod1          offset for prev item that needs to be added
 * @param rounded_length the rounded length of array
 * @param active_int     the number of threads that is working
 */
__global__ void cuda_es_upsweep_varblock(int* device_result,
        int twod, int twod1,
        int rounded_length,
        int active_int) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int dst_index = (i + 1) * twod1 - 1;
    unsigned int src_index = dst_index - twod;
    unsigned int unsigned_rounded_length = rounded_length;
    if (dst_index < unsigned_rounded_length) {
        device_result[dst_index] += device_result[src_index];
    }
}

/**
 * Down sweep process of naive scan algorithm by adjustting working threads
 * @param device_result  the input array and output array
 * @param twod           offset to find the next source index
 * @param twod1          offset for prev item that needs to be added
 * @param rounded_length the rounded length of array
 */
__global__ void cuda_es_downsweep_varblock(int* device_result,
        int twod, int twod1,
        int rounded_length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int dst_index = (i + 1) * twod1 - 1;
    unsigned int src_index = dst_index - twod;
    unsigned int rounded_length_unisigned = rounded_length;

    if (dst_index < rounded_length_unisigned) {
        int t = device_result[src_index];
        device_result[src_index] = device_result[dst_index];
        device_result[dst_index] += t;
    }
}

/**
 * Naive scan algorithm improved by adjustting working threads
 * @param device_start  the input array
 * @param length        the actual length of input array
 * @param device_result the output array
 */
void exclusive_scan_varblock(int* device_start, int length, int* device_result) {
    int threadsPerBlock = THREADS_PER_BLOCK;
    int rounded_length = nextPow2(length);
    int blockPerGrid = std::max(1, rounded_length / threadsPerBlock / 2);
    int active_int = rounded_length / 2;
    for (int twod = 1; twod < rounded_length / 2; twod *= 2) {
        int twod1 = twod * 2;
        // This is intended to go over than 80 characters for better readness
        cuda_es_upsweep_varblock <<< blockPerGrid, threadsPerBlock>>>(device_result, twod, twod1, rounded_length, active_int);
        cudaCheckError(cudaThreadSynchronize());
        blockPerGrid = std::max(1, blockPerGrid / 2);
        active_int /= 2;
    }
    cudaError_t cudaError =
        cudaMemset(device_result + rounded_length - 1, 0, sizeof(int));
    cudaCheckError(cudaError);

    int active_item = 1;
    for (int twod = rounded_length / 2; twod >= 1; twod /= 2) {
        int twod1 = twod * 2;
        // This is intended to go over than 80 characters for better readness
        cuda_es_downsweep_varblock <<< blockPerGrid, threadsPerBlock>>>(device_result, twod, twod1, rounded_length);
        cudaCheckError(cudaThreadSynchronize());
        active_item *= 2;
        blockPerGrid = std::max(1, active_item / THREADS_PER_BLOCK);
    }
}

/**
 * Up sweep process of naive scan algorithm
 * @param device_result  the input array and output array
 * @param twod           offset to find the next source index
 * @param twod1          offset for prev item that needs to be added
 * @param rounded_length the rounded length of array
 */
__global__ void cuda_es_upsweep(int* device_result, int twod, int twod1) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if ((i + 1) % twod1 != 0) {
        return;
    }
    device_result[i] += device_result[i - twod];
}

/**
 * Down sweep process of naive scan algorithm
 * @param device_result  the input array and output array
 * @param twod           offset to find the next source index
 * @param twod1          offset for prev item that needs to be added
 */
__global__ void cuda_es_downsweep(int* device_result, int twod, int twod1) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if ((i + 1) % twod1 != 0) {
        return;
    }
    int t = device_result[i - twod];
    device_result[i - twod] = device_result[i];
    device_result[i] += t;
}

/**
 * Naive scan algorithm
 * @param device_start  the input array
 * @param length        the actual length of input array
 * @param device_result the output array
 */
void exclusive_scan_naive(int* device_start, int length, int* device_result) {
    int threadsPerBlock = THREADS_PER_BLOCK;
    int rounded_length = nextPow2(length);
    int blockPerGrid = std::max(1, rounded_length / threadsPerBlock / 2);

    for (int twod = 1; twod < rounded_length / 2; twod *= 2) {
        int twod1 = twod * 2;
        cuda_es_upsweep <<< blockPerGrid, threadsPerBlock>>>(device_result, twod, twod1);
        cudaCheckError(cudaThreadSynchronize());
    }
    cudaCheckError(cudaMemset(device_result + rounded_length - 1, 0, sizeof(int)));

    for (int twod = rounded_length / 2; twod >= 1; twod /= 2) {
        int twod1 = twod * 2;
        // This is intended to go over than 80 characters for better readness
        cuda_es_downsweep <<< blockPerGrid, threadsPerBlock>>>(device_result, twod, twod1);
        cudaCheckError(cudaThreadSynchronize());
    }
}

/**
 * This warpper chooes most efficcient algorithm for scan, if the input array is
 * not so big, there is no need to pay more overhead on a short-length array.
 * @param device_input  the input array
 * @param real_len      the actual length of input array
 * @param device_result the output array
 */
void exclusive_scan(int* device_input, int real_len, int* device_result) {
    if (real_len > 1000000) {
        int rounded_length = std::max(nextPow2(real_len),
                                      MAX_DATA_LENGTH_PER_BLOCK);
        exclusive_scan_sharedmem(device_result, rounded_length, real_len);
    } else {
        exclusive_scan_varblock(device_input, real_len, device_result);
    }
}

/**
 * Scan funtion entry that is provided by instructor
 * @param  inarray     the input of array
 * @param  end         the end ptr of array
 * @param  resultarray the result of array
 * @return             time used by scaning.
 */
double cudaScan(int* inarray, int* end, int* resultarray) {
    int* device_result;
    int* device_input;

    /**
     *  In order to better handle the situation when the array length is shorter
     *  than MAX_DATA_LENGTH_PER_BLOCK
     */
    int rounded_length = std::max(nextPow2(end - inarray),
                                  MAX_DATA_LENGTH_PER_BLOCK);
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMemset(device_result, 0, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);
    cudaMemset(device_result, 0, rounded_length * sizeof(int));
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    exclusive_scan(device_input, end - inarray, device_result);
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaFree(device_result);
    cudaFree(device_input);
    return overallDuration;
}

/**
 * Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself
 * @param  inarray     the input of array
 * @param  end         the end ptr of array
 * @param  resultarray the result of array
 * @return             time used by scaning
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}

/**
 * Find whether the value of each item's right is equal to itself
 * @param is_repeated  the result array to indicate whether there is a repeat
 * @param device_input the input array
 * @param length       the length of the input array
 */
__global__ void check_right_repeat(int *is_repeated,
                                   int *device_input, int length) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < length - 1)
        is_repeated[id] = (device_input[id] == device_input[id + 1]);
    else
        is_repeated[id] = 0;
}

/**
 * Find the repeat index wit the result from scan
 * @param is_repeated_prefix_sum The prefix sum array from scan
 * @param device_output          the output array
 * @param length                 the length of the outout array
 */
__global__ void get_repeat_index(int *is_repeated_prefix_sum,
                                 int *device_output,
                                 int length) {
    unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < length - 1) {
        unsigned int output_index = is_repeated_prefix_sum[id];
        if (output_index < is_repeated_prefix_sum[id + 1]) {
            device_output[output_index] = id;
        }
    }
}

/**
 * Find repeats inside array with scan
 * @param  device_input  the input array
 * @param  length        the length of input array
 * @param  device_output the output array
 * @return               the length of output array
 */
int find_repeats(int *device_input, int length, int *device_output) {
    int threadsPerBlock = THREADS_PER_BLOCK_REPEAT;
    int rd_len = nextPow2(length);
    int blockPerGrid = std::max(1, nextPow2(length) / (threadsPerBlock));
    int* is_repeated;
    cudaMalloc((void **)&is_repeated, length * sizeof(int));
    check_right_repeat <<< blockPerGrid, threadsPerBlock>>>(is_repeated,
            device_input,
            length);
    int *is_repeated_prefix_sum;
    int rounded_length = std::max(rd_len, THREADS_MAX_PER_BLOCK);
    cudaMalloc((void **)&is_repeated_prefix_sum, sizeof(int) * rounded_length);
    cudaMemset(is_repeated_prefix_sum, 0, rounded_length * sizeof(int));
    cudaMemcpy(is_repeated_prefix_sum, is_repeated, length * sizeof(int),
               cudaMemcpyHostToDevice);
    exclusive_scan_sharedmem(is_repeated_prefix_sum, rounded_length, length);
    // This is intended to go over than 80 characters for better readness
    get_repeat_index <<< blockPerGrid, threadsPerBlock>>>(is_repeated_prefix_sum, device_output, length);
    cudaThreadSynchronize();
    int device_output_len = 0;
    cudaMemcpy(&device_output_len,
               is_repeated_prefix_sum + length - 1,
               sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaFree(is_repeated_prefix_sum);
    return device_output_len;
}

/**
 * Timing wrapper around find_repeats. You should not modify this function.
 * @param  input         the input array
 * @param  length        the length of input array
 * @param  output        the output array
 * @param  output_length the length of output array
 * @return               time used by finding repeats
 */
double cudaFindRepeats(int *input, int length,
                       int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    int result = find_repeats(device_input, length, device_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}

/**
 * Print video card information
 */
void printCudaInfo() {
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
