// FMA Latency Measurement Kernels
//
// This file contains various CUDA kernels to measure the latency of
// fused multiply-add (FMA) operations under different execution patterns:
// - Basic latency measurement
// - Interleaved execution (ILP)
// - Non-interleaved execution (sequential chains)

#include <cuda_runtime.h>
#include <iostream>

using data_type = float;

// Inline assembly macro to read GPU cycle counter
#define clock_cycle() \
    ({ \
        unsigned long long ret; \
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(ret)); \
        ret; \
    })

////////////////////////////////////////////////////////////////////////////////
// Basic FMA Latency

__global__ void
fma_latency(data_type *n, unsigned long long *d_start, unsigned long long *d_end) {
    unsigned long long start_time = clock_cycle();
    unsigned long long end_time = clock_cycle();
    data_type x = *n;
    // Memory fence to ensure that the reads are done.
    __threadfence();
    start_time = clock_cycle();

    /// <--- /your code here --->

    end_time = clock_cycle();

    *n = x;
    *d_start = start_time;
    *d_end = end_time;
}

////////////////////////////////////////////////////////////////////////////////
// FMA Latency + Instruction Level Parallelism (Interleaved)

__global__ void fma_latency_interleaved(
    data_type *n,
    unsigned long long *d_start,
    unsigned long long *d_end) {
    unsigned long long start_time = clock_cycle();
    unsigned long long end_time = clock_cycle();

    data_type x = *n;
    data_type y = *n;
    // Memory fence to ensure that the reads are done.
    __threadfence();

    start_time = clock_cycle();

    /// <--- /your code here --->

    end_time = clock_cycle();

    *n = x + y;
    *d_start = start_time;
    *d_end = end_time;
}

////////////////////////////////////////////////////////////////////////////////
// FMA Latency + Sequential Execution (No Interleaving)

__global__ void fma_latency_no_interleave(
    data_type *n,
    unsigned long long *d_start,
    unsigned long long *d_end) {

    unsigned long long start_time = clock_cycle();
    unsigned long long end_time = clock_cycle();

    data_type x = *n;
    data_type y = *n;
    // Memory fence to ensure that the reads are done.
    __threadfence();

    start_time = clock_cycle();

    /// <--- /your code here --->

    end_time = clock_cycle();

    *n = x + y;
    *d_start = start_time;
    *d_end = end_time;
}

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

// CUDA error checking macro
#define CUDA_CHECK(x) \
    do { \
        cudaError_t err = (x); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error " << static_cast<int>(err) << " (" \
                      << cudaGetErrorString(err) << ") at " << __FILE__ << ":" \
                      << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// Macro to run kernel and print timing results.
#define run_kernel_and_print(kernel, d_n, d_start, d_end) \
    do { \
        unsigned long long h_time_start = 0ull, h_time_end = 0ull; \
        data_type result = 0.0f; \
\
        kernel<<<1, 1>>>(d_n, d_start, d_end); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
        CUDA_CHECK(cudaMemcpy( \
            &h_time_start, \
            d_start, \
            sizeof(unsigned long long), \
            cudaMemcpyDeviceToHost)); \
        CUDA_CHECK(cudaMemcpy( \
            &h_time_end, \
            d_end, \
            sizeof(unsigned long long), \
            cudaMemcpyDeviceToHost)); \
        CUDA_CHECK(cudaMemcpy(&result, d_n, sizeof(data_type), cudaMemcpyDeviceToHost)); \
\
        std::cout << "Latency of " << #kernel \
                  << " code snippet = " << (h_time_end - h_time_start) << " cycles" \
                  << std::endl; \
    } while (0)

int main() {
    data_type *d_n = nullptr;
    unsigned long long *d_start = nullptr;
    unsigned long long *d_end = nullptr;

    data_type host_val = 4.0f;
    unsigned long long host_init_time = 0ull;

    CUDA_CHECK(cudaMalloc(&d_n, sizeof(data_type)));
    CUDA_CHECK(cudaMalloc(&d_start, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_end, sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpy(d_n, &host_val, sizeof(data_type), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_start,
        &host_init_time,
        sizeof(unsigned long long),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_end,
        &host_init_time,
        sizeof(unsigned long long),
        cudaMemcpyHostToDevice));

    run_kernel_and_print(fma_latency, d_n, d_start, d_end);
    run_kernel_and_print(fma_latency_interleaved, d_n, d_start, d_end);
    run_kernel_and_print(fma_latency_no_interleave, d_n, d_start, d_end);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_n));
    CUDA_CHECK(cudaFree(d_start));
    CUDA_CHECK(cudaFree(d_end));

    return 0;
}