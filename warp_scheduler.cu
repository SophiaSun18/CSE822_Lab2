// Optional arguments:
//  -b <num_threads_per_block>

#include <cstdint>
#include <cuda_runtime.h>

constexpr uint32_t default_num_ops = 263 * 20;
constexpr uint32_t default_tile_size = 32;
constexpr uint32_t default_num_threads_per_block = 64;

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

////////////////////////////////////////////////////////////////////////////////
// FMA Tail Latency Kernel

template <int NumOps, int Tile>
__global__ void
tail_latency_fma(const float *__restrict__ in, float *__restrict__ out, int N) {
    int index = blockIdx.x * blockDim.x * Tile + threadIdx.x * Tile;

    for (int i = 0; i < Tile; ++i) {
        float tmp = in[index + i];
#pragma unroll
        for (int op = 0; op < NumOps; ++op) {
            tmp += tmp * 3.0f;
        }
        out[index + i] = tmp;
    }
}

void launch_tail_latency_fma(
    const float *in,
    float *out,
    int N,
    uint32_t tile_size,
    uint32_t num_threads_per_block) {
    uint32_t block_size = default_tile_size * num_threads_per_block;
    dim3 block(num_threads_per_block);
    dim3 grid(N / block_size);

    tail_latency_fma<default_num_ops, default_tile_size><<<grid, block>>>(in, out, N);
}

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

// Command-line arguments parser.
int ParseArgsAndMakeSpec(int argc, char *argv[], uint32_t *num_threads_per_block) {

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                *num_threads_per_block = atoi(argv[++i]);
            } else {
                std::cerr << "Error: No value specified for -b" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Unknown flag: " << argv[i] << std::endl;
            return 1;
        }
    }

    return 0;
}

// Benchmarking macros and configuration.
#define BENCHPRESS(func, kNumOfOuterIterations, kNumOfInnerIterations, ...) \
    do { \
        std::vector<double> times(kNumOfOuterIterations); \
        for (size_t i = 0; i < kNumOfOuterIterations; ++i) { \
            auto start = std::chrono::high_resolution_clock::now(); \
            for (size_t j = 0; j < kNumOfInnerIterations; ++j) { \
                func(__VA_ARGS__); \
            } \
            CUDA_CHECK(cudaDeviceSynchronize()); \
            auto end = std::chrono::high_resolution_clock::now(); \
            times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start) \
                           .count() / \
                kNumOfInnerIterations; \
        } \
        std::sort(times.begin(), times.end()); \
        benchmark_time_ns = times[0]; \
    } while (0)

// AUX CUDA check functions.
void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

// Main function.
int main(int argc, char *argv[]) {
    // Get tail latency spec.
    uint32_t num_threads_per_block = default_num_threads_per_block;

    if (ParseArgsAndMakeSpec(argc, argv, &num_threads_per_block))
        return -1;

    std::vector<int> n_multipliers = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                      25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                      37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48};

    uint32_t block_size = default_tile_size * num_threads_per_block;
    double benchmark_time_ns = 0.0;

    std::cout << "Warps, FLOPS" << std::endl;
    for (int multiplier : n_multipliers) {
        int N = block_size * 48 * multiplier;

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_in, 1, N * sizeof(float)));

        BENCHPRESS(
            launch_tail_latency_fma,
            3,
            10,
            d_in,
            d_out,
            N,
            default_tile_size,
            num_threads_per_block);

        double time_in_seconds = benchmark_time_ns / 1e9;
        double flops = ((double)N * default_num_ops * 2) / time_in_seconds;
        uint32_t warps = (N / block_size) * (num_threads_per_block / 32);

        std::cout << warps << ", " << flops << std::endl;

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    return 0;
}