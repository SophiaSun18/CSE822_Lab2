// Optional arguments:
//  -r <img_size>
//  -b <max iterations>
//  -i <implementation: {"scalar", "vector", "vector_ilp", "vector_multicore",
//  "vector_multicore_multithread", "vector_multicore_multithread_ilp", "all"}>

#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <pthread.h>

constexpr float window_zoom = 1.0 / 10000.0f;
constexpr float window_x = -0.743643887 - 0.5 * window_zoom;
constexpr float window_y = 0.131825904 - 0.5 * window_zoom;
constexpr uint32_t default_max_iters = 2000;
const int VECTOR_SIZE = 16;
const int UNROLL_FACTOR = 4;
int NUM_THREAD_SINGLE = 8;
int NUM_THREAD_PER_CORE = 2;
int NUM_CORE = 10;

// CPU Scalar Mandelbrot set generation.
// Based on the "optimized escape time algorithm" in
// https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set
void mandelbrot_cpu_scalar(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    for (uint64_t i = 0; i < img_size; ++i) {
        for (uint64_t j = 0; j < img_size; ++j) {
            float cx = (float(j) / float(img_size)) * window_zoom + window_x;
            float cy = (float(i) / float(img_size)) * window_zoom + window_y;

            float x2 = 0.0f;
            float y2 = 0.0f;
            float w = 0.0f;
            uint32_t iters = 0;
            while (x2 + y2 <= 4.0f && iters < max_iters) {
                float x = x2 - y2 + cx;
                float y = w - (x2 + y2) + cy;
                x2 = x * x;
                y2 = y * y;
                float z = x + y;
                w = z * z;
                ++iters;
            }

            // Write result.
            out[i * img_size + j] = iters;
        }
    }
}

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

/// <--- your code here --->

// OPTIONAL: Uncomment this block to include your CPU vector implementation
// from Lab 1 for easy comparison.
//
// (If you do this, you'll need to update your code to use the new constants
// 'window_zoom', 'window_x', and 'window_y'.)

#define HAS_VECTOR_IMPL // <~~ keep this line if you want to benchmark the vector kernel!

////////////////////////////////////////////////////////////////////////////////
// Vector

void mandelbrot_cpu_vector_256(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    const int vector_size = 8;
    const float scalar = window_zoom / float(img_size);
    const __m256 v_scale = _mm256_set1_ps(scalar);
    const __m256 v_wx = _mm256_set1_ps(window_x);
    const __m256 r_4 = _mm256_set1_ps(4.0f);

    for (uint64_t i = 0; i < img_size; i++) {
        float cy_scalar = float(i) * scalar + window_y;
        __m256 cy = _mm256_set1_ps(cy_scalar);

        for (uint64_t j = 0; j < img_size; j += vector_size) {
            // Get the plane coordinate X for the image pixel.
            __m256 cx = _mm256_set_ps(
                float(j + 7), float(j + 6), float(j + 5), float(j + 4),
                float(j + 3), float(j + 2), float(j + 1), float(j));
            cx = _mm256_add_ps(_mm256_mul_ps(cx, v_scale), v_wx);

            // Innermost loop: start the recursion from z = 0.
            __m256 x2 = _mm256_set1_ps(0.0f);
            __m256 y2 = _mm256_set1_ps(0.0f);
            __m256 w = _mm256_set1_ps(0.0f);
            __m256i iters = _mm256_set1_epi32(0);

            for (uint32_t k = 0; k < max_iters; k++) {
                // Calculate x2 + y2 and check if sum <= 4.0f to generate new mask.
                __m256 sum = _mm256_add_ps(x2, y2);
                __m256 mask_ps = _mm256_cmp_ps(sum, r_4, _CMP_LE_OQ);
                __m256i mask = _mm256_castps_si256(mask_ps);

                // If all elements are no longer active, break.
                if (_mm256_testz_si256(mask, mask)) break;

                __m256 x_new = _mm256_add_ps(_mm256_sub_ps(x2, y2), cx);
                __m256 y_new = _mm256_add_ps(_mm256_sub_ps(w, sum), cy);
                __m256 z_new = _mm256_add_ps(x_new, y_new);
                __m256 w_new = _mm256_mul_ps(z_new, z_new);
                __m256 x2_new = _mm256_mul_ps(x_new, x_new);
                __m256 y2_new = _mm256_mul_ps(y_new, y_new);

                // Update x2 and y2 according to the mask.
                x2 = _mm256_blendv_ps(x2, x2_new, mask_ps);
                y2 = _mm256_blendv_ps(y2, y2_new, mask_ps);
                w  = _mm256_blendv_ps(w, w_new, mask_ps);

                // Update iters based on the number of active elements.
                __m256i one = _mm256_set1_epi32(1);
                iters = _mm256_add_epi32(iters, _mm256_and_si256(one, mask));
            }

            // Write result.
            _mm256_storeu_si256((__m256i*)&out[i*img_size + j], iters);
        }
    }
}

void mandelbrot_cpu_vector_512(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    const float scalar = window_zoom / float(img_size);
    const __m512 v_scale = _mm512_set1_ps(scalar);
    const __m512 v_wx = _mm512_set1_ps(window_x);
    const __m512 r_4 = _mm512_set1_ps(4.0f);

    for (uint64_t i = 0; i < img_size; i++) {
        float cy_scalar = float(i) * scalar + window_y;
        __m512 cy = _mm512_set1_ps(cy_scalar);

        for (uint64_t j = 0; j < img_size; j+=VECTOR_SIZE) {
            // Get the plane coordinate X for the image pixel.
            __m512 cx = _mm512_set_ps(
                float(j + 15), float(j + 14), float(j + 13), float(j + 12),
                float(j + 11), float(j + 10), float(j + 9), float(j + 8),
                float(j + 7), float(j + 6), float(j + 5), float(j + 4),
                float(j + 3), float(j + 2), float(j + 1), float(j));
            cx = _mm512_add_ps(_mm512_mul_ps(cx, v_scale), v_wx);

            // Innermost loop: start the recursion from z = 0.
            __m512 x2 = _mm512_set1_ps(0.0f);
            __m512 y2 = _mm512_set1_ps(0.0f);
            __m512 w = _mm512_set1_ps(0.0f);
            __m512i iters = _mm512_set1_epi32(0);
            __mmask16 active = 0xFFFF;

            for (uint32_t k = 0; k < max_iters; k++) {
                // Calculate x2 + y2 and check if sum <= 4.0f to generate new mask.
                __m512 sum = _mm512_add_ps(x2, y2);
                active = _mm512_cmp_ps_mask(sum, r_4, _CMP_LE_OQ);

                // Update iters based on current active bits.
                iters = _mm512_mask_add_epi32(iters, active, iters, _mm512_set1_epi32(1));
                if (active == 0) break;

                __m512 x = _mm512_add_ps(_mm512_sub_ps(x2, y2), cx);
                __m512 y = _mm512_add_ps(_mm512_sub_ps(w, sum), cy);
                x2 = _mm512_mul_ps(x, x);
                y2 = _mm512_mul_ps(y, y);
                __m512 z = _mm512_add_ps(x, y);
                w = _mm512_mul_ps(z, z);
            }

            // Write result.
            _mm512_storeu_si512((__m512i*)&out[i * img_size + j], iters);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Vector + ILP

void mandelbrot_cpu_vector_ilp(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    const float scalar = window_zoom / float(img_size);
    const __m512 v_scale = _mm512_set1_ps(scalar);
    const __m512 v_wx = _mm512_set1_ps(window_x);
    const __m512 r_4 = _mm512_set1_ps(4.0f);

    for (uint64_t i = 0; i < img_size; i+=UNROLL_FACTOR) {
        for (uint64_t j = 0; j < img_size; j+=VECTOR_SIZE) {
            __m512 cx = _mm512_set_ps(
                float(j + 15), float(j + 14), float(j + 13), float(j + 12),
                float(j + 11), float(j + 10), float(j + 9), float(j + 8),
                float(j + 7), float(j + 6), float(j + 5), float(j + 4),
                float(j + 3), float(j + 2), float(j + 1), float(j));
            cx = _mm512_add_ps(_mm512_mul_ps(cx, v_scale), v_wx);

            __m512 vec_cy[UNROLL_FACTOR];
            __m512 vec_x2[UNROLL_FACTOR];
            __m512 vec_y2[UNROLL_FACTOR];
            __m512 vec_w[UNROLL_FACTOR];
            __m512i vec_iters[UNROLL_FACTOR];
            __mmask16 active[UNROLL_FACTOR];

            #pragma GCC unroll UNROLL_FACTOR
            for (int k = 0; k < UNROLL_FACTOR; k++) {
                float cy_scalar = float(i + k) * scalar + window_y;
                vec_cy[k] = _mm512_set1_ps(cy_scalar);
                vec_x2[k] = _mm512_set1_ps(0.0f);
                vec_y2[k] = _mm512_set1_ps(0.0f);
                vec_w[k] = _mm512_set1_ps(0.0f);
                vec_iters[k] = _mm512_set1_epi32(0);
                active[k] = 0xFFFF;
            }

            for (uint32_t m = 0; m < max_iters; m++) {
                #pragma GCC unroll UNROLL_FACTOR
                for (int k = 0; k < UNROLL_FACTOR; k++) {
                    __m512 sum = _mm512_add_ps(vec_x2[k], vec_y2[k]);
                    vec_iters[k] = _mm512_mask_add_epi32(vec_iters[k], active[k], vec_iters[k], _mm512_set1_epi32(1));

                    __m512 x = _mm512_add_ps(_mm512_sub_ps(vec_x2[k], vec_y2[k]), cx);
                    __m512 y = _mm512_add_ps(_mm512_sub_ps(vec_w[k], sum), vec_cy[k]);
                    vec_x2[k] = _mm512_mul_ps(x, x);
                    vec_y2[k] = _mm512_mul_ps(y, y);
                    __m512 z = _mm512_add_ps(x, y);
                    vec_w[k] = _mm512_mul_ps(z, z);
                }
                #pragma GCC unroll UNROLL_FACTOR
                for (int k = 0; k < UNROLL_FACTOR; k++) {
                    active[k] = _mm512_cmp_ps_mask(_mm512_add_ps(vec_x2[k], vec_y2[k]), r_4, _CMP_LE_OQ);
                }
                if (active[0] == 0 && active[1] == 0 && active[2] == 0 && active[3] == 0) {
                    break;
                }
            }

            #pragma GCC unroll UNROLL_FACTOR
            for (int k = 0; k < UNROLL_FACTOR; k++) {
                _mm512_storeu_si512((__m512i*)&out[(i + k) * img_size + j], vec_iters[k]);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core

typedef struct {
    uint32_t img_size;
    uint32_t max_iters;
    uint32_t thread_id;
    uint32_t *out;
} thread_args;

void mandelbrot_cpu_vector_thread(uint32_t img_size, uint32_t max_iters, uint32_t thread_id, uint32_t *out) {
    const float scalar = window_zoom / float(img_size);
    const __m512 v_scale = _mm512_set1_ps(scalar);
    const __m512 v_wx = _mm512_set1_ps(window_x);
    const __m512 r_4 = _mm512_set1_ps(4.0f);

    for (uint64_t i = thread_id; i < img_size; i+=NUM_THREAD_SINGLE) {
        // Compute cy_scalar repeatedly, which is cheaper than computing cx.
        float cy_scalar = float(i) * scalar + window_y;
        __m512 cy = _mm512_set1_ps(cy_scalar);

        for (uint64_t j = 0; j + VECTOR_SIZE <= img_size; j+=VECTOR_SIZE) {
            // Get the plane coordinate X for the image pixel.
            // cx can be shared by all current rows.
            __m512 cx = _mm512_set_ps(
                float(j + 15), float(j + 14), float(j + 13), float(j + 12),
                float(j + 11), float(j + 10), float(j + 9), float(j + 8),
                float(j + 7), float(j + 6), float(j + 5), float(j + 4),
                float(j + 3), float(j + 2), float(j + 1), float(j));
            cx = _mm512_add_ps(_mm512_mul_ps(cx, v_scale), v_wx);

            // Innermost loop: start the recursion from z = 0.
            __m512 x2 = _mm512_set1_ps(0.0f);
            __m512 y2 = _mm512_set1_ps(0.0f);
            __m512 w = _mm512_set1_ps(0.0f);
            __m512i iters = _mm512_set1_epi32(0);
            __mmask16 active = 0xFFFF;

            for (uint32_t k = 0; k < max_iters; k++) {
                // Calculate x2 + y2 and check if sum <= 4.0f to generate new mask.
                __m512 sum = _mm512_add_ps(x2, y2);
                active = _mm512_cmp_ps_mask(sum, r_4, _CMP_LE_OQ);

                // Update iters based on current active bits.
                iters = _mm512_mask_add_epi32(iters, active, iters, _mm512_set1_epi32(1));
                if (active == 0) break;

                __m512 x = _mm512_add_ps(_mm512_sub_ps(x2, y2), cx);
                __m512 y = _mm512_add_ps(_mm512_sub_ps(w, sum), cy);
                x2 = _mm512_mul_ps(x, x);
                y2 = _mm512_mul_ps(y, y);
                __m512 z = _mm512_add_ps(x, y);
                w = _mm512_mul_ps(z, z);
            }

            // Write result.
            _mm512_storeu_si512((__m512i*)&out[i * img_size + j], iters);
        }
    }
}

void* mandelbrot_cpu_vector_thread_wrapper(void* arg) {
    thread_args* args = (thread_args*) arg;
    mandelbrot_cpu_vector_thread(
        args->img_size,
        args->max_iters,
        args->thread_id,
        args->out
    );
    return NULL;
}

void mandelbrot_cpu_vector_multicore(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    pthread_t threads[NUM_THREAD_SINGLE];
    thread_args args[NUM_THREAD_SINGLE];

    for (int i = 0; i < NUM_THREAD_SINGLE; ++i) {
        args[i].img_size = img_size;
        args[i].max_iters = max_iters;
        args[i].thread_id = i;
        args[i].out = out;
        pthread_create(&threads[i], NULL, mandelbrot_cpu_vector_thread_wrapper, &args[i]);
    }

    for (int j = 0; j < NUM_THREAD_SINGLE; ++j) {
        pthread_join(threads[j], NULL);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core + Multi-thread-per-core

void mandelbrot_cpu_vector_multithread(uint32_t img_size, uint32_t max_iters, uint32_t thread_id, uint32_t *out) {
    const float scalar = window_zoom / float(img_size);
    const __m512 v_scale = _mm512_set1_ps(scalar);
    const __m512 v_wx = _mm512_set1_ps(window_x);
    const __m512 r_4 = _mm512_set1_ps(4.0f);

    int TOTAL_THREAD = NUM_THREAD_PER_CORE * NUM_CORE;

    for (uint64_t i = thread_id; i < img_size; i+=TOTAL_THREAD) {
        // Compute cy_scalar repeatedly, which is cheaper than computing cx.
        float cy_scalar = float(i) * scalar + window_y;
        __m512 cy = _mm512_set1_ps(cy_scalar);

        for (uint64_t j = 0; j + VECTOR_SIZE <= img_size; j+=VECTOR_SIZE) {
            // Get the plane coordinate X for the image pixel.
            // cx can be shared by all current rows.
            __m512 cx = _mm512_set_ps(
                float(j + 15), float(j + 14), float(j + 13), float(j + 12),
                float(j + 11), float(j + 10), float(j + 9), float(j + 8),
                float(j + 7), float(j + 6), float(j + 5), float(j + 4),
                float(j + 3), float(j + 2), float(j + 1), float(j));
            cx = _mm512_add_ps(_mm512_mul_ps(cx, v_scale), v_wx);

            // Innermost loop: start the recursion from z = 0.
            __m512 x2 = _mm512_set1_ps(0.0f);
            __m512 y2 = _mm512_set1_ps(0.0f);
            __m512 w = _mm512_set1_ps(0.0f);
            __m512i iters = _mm512_set1_epi32(0);
            __mmask16 active = 0xFFFF;

            for (uint32_t k = 0; k < max_iters; k++) {
                // Calculate x2 + y2 and check if sum <= 4.0f to generate new mask.
                __m512 sum = _mm512_add_ps(x2, y2);
                active = _mm512_cmp_ps_mask(sum, r_4, _CMP_LE_OQ);

                // Update iters based on current active bits.
                iters = _mm512_mask_add_epi32(iters, active, iters, _mm512_set1_epi32(1));
                if (active == 0) break;

                __m512 x = _mm512_add_ps(_mm512_sub_ps(x2, y2), cx);
                __m512 y = _mm512_add_ps(_mm512_sub_ps(w, sum), cy);
                x2 = _mm512_mul_ps(x, x);
                y2 = _mm512_mul_ps(y, y);
                __m512 z = _mm512_add_ps(x, y);
                w = _mm512_mul_ps(z, z);
            }

            // Write result.
            _mm512_storeu_si512((__m512i*)&out[i*img_size + j], iters);
        }
    }
}

void* mandelbrot_cpu_vector_multithread_wrapper(void* arg) {
    thread_args* args = (thread_args*) arg;
    mandelbrot_cpu_vector_multithread(
        args->img_size,
        args->max_iters,
        args->thread_id,
        args->out
    );
    return NULL;
}

void mandelbrot_cpu_vector_multicore_multithread(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    int TOTAL_THREAD = NUM_THREAD_PER_CORE * NUM_CORE;

    pthread_t threads[TOTAL_THREAD];
    thread_args args[TOTAL_THREAD];

    for (int i = 0; i < TOTAL_THREAD; ++i) {
        args[i].img_size = img_size;
        args[i].max_iters = max_iters;
        args[i].thread_id = i;
        args[i].out = out;
        pthread_create(&threads[i], NULL, mandelbrot_cpu_vector_multithread_wrapper, &args[i]);
    }

    for (int j = 0; j < TOTAL_THREAD; ++j) {
        pthread_join(threads[j], NULL);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core + Multi-thread-per-core + ILP

void mandelbrot_cpu_vector_multicore_multithread_ilp_thread(uint32_t img_size, uint32_t max_iters, uint32_t thread_id, uint32_t *out) {
    const float scalar = window_zoom / float(img_size);
    const __m512 v_scale = _mm512_set1_ps(scalar);
    const __m512 v_wx = _mm512_set1_ps(window_x);
    const __m512 r_4 = _mm512_set1_ps(4.0f);

    int TOTAL_THREAD = NUM_THREAD_PER_CORE * NUM_CORE;
    
    for (uint64_t base_i = thread_id * UNROLL_FACTOR; base_i < img_size; base_i+=UNROLL_FACTOR*TOTAL_THREAD) {
        for (uint64_t j = 0; j < img_size; j+=VECTOR_SIZE) {
            __m512 cx = _mm512_set_ps(
                float(j + 15), float(j + 14), float(j + 13), float(j + 12),
                float(j + 11), float(j + 10), float(j + 9), float(j + 8),
                float(j + 7), float(j + 6), float(j + 5), float(j + 4),
                float(j + 3), float(j + 2), float(j + 1), float(j));
            cx = _mm512_add_ps(_mm512_mul_ps(cx, v_scale), v_wx);

            __m512 vec_cy[UNROLL_FACTOR];
            __m512 vec_x2[UNROLL_FACTOR];
            __m512 vec_y2[UNROLL_FACTOR];
            __m512 vec_w[UNROLL_FACTOR];
            __m512i vec_iters[UNROLL_FACTOR];
            __mmask16 active[UNROLL_FACTOR];

            #pragma GCC unroll UNROLL_FACTOR
            for (int k = 0; k < UNROLL_FACTOR; k++) {
                uint64_t i = base_i + k;
                float cy_scalar = float(i) * scalar + window_y;
                vec_cy[k] = _mm512_set1_ps(cy_scalar);
                vec_x2[k] = _mm512_set1_ps(0.0f);
                vec_y2[k] = _mm512_set1_ps(0.0f);
                vec_w[k] = _mm512_set1_ps(0.0f);
                vec_iters[k] = _mm512_set1_epi32(0);
                active[k] = 0xFFFF;
            }

            for (uint32_t m = 0; m < max_iters; m++) {
                #pragma GCC unroll UNROLL_FACTOR
                for (int k = 0; k < UNROLL_FACTOR; k++) {
                    __m512 sum = _mm512_add_ps(vec_x2[k], vec_y2[k]);
                    vec_iters[k] = _mm512_mask_add_epi32(vec_iters[k], active[k], vec_iters[k], _mm512_set1_epi32(1));

                    __m512 x = _mm512_add_ps(_mm512_sub_ps(vec_x2[k], vec_y2[k]), cx);
                    __m512 y = _mm512_add_ps(_mm512_sub_ps(vec_w[k], sum), vec_cy[k]);
                    vec_x2[k] = _mm512_mul_ps(x, x);
                    vec_y2[k] = _mm512_mul_ps(y, y);
                    __m512 z = _mm512_add_ps(x, y);
                    vec_w[k] = _mm512_mul_ps(z, z);
                }
                #pragma GCC unroll UNROLL_FACTOR
                for (int k = 0; k < UNROLL_FACTOR; k++) {
                    active[k] = _mm512_cmp_ps_mask(_mm512_add_ps(vec_x2[k], vec_y2[k]), r_4, _CMP_LE_OQ);
                }
                if (active[0] == 0 && active[1] == 0 && active[2] == 0 && active[3] == 0) {
                    break;
                }
            }

            #pragma GCC unroll UNROLL_FACTOR
            for (int k = 0; k < UNROLL_FACTOR; k++) {
                uint64_t i = base_i + k;
                _mm512_storeu_si512((__m512i*)&out[i * img_size + j], vec_iters[k]);
            }
        }
    }
}

void* mandelbrot_cpu_vector_multicore_multithread_ilp_thread_wrapper(void* arg) {
    thread_args* args = (thread_args*) arg;
    mandelbrot_cpu_vector_multicore_multithread_ilp_thread(
        args->img_size,
        args->max_iters,
        args->thread_id,
        args->out
    );
    return NULL;
}

void mandelbrot_cpu_vector_multicore_multithread_ilp(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    int TOTAL_THREAD = NUM_THREAD_PER_CORE * NUM_CORE;

    pthread_t threads[TOTAL_THREAD];
    thread_args args[TOTAL_THREAD];

    for (int i = 0; i < TOTAL_THREAD; ++i) {
        args[i].img_size = img_size;
        args[i].max_iters = max_iters;
        args[i].thread_id = i;
        args[i].out = out;
        pthread_create(&threads[i], NULL, mandelbrot_cpu_vector_multicore_multithread_ilp_thread_wrapper, &args[i]);
    }

    for (int j = 0; j < TOTAL_THREAD; ++j) {
        pthread_join(threads[j], NULL);
    }
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <vector>

// Useful functions and structures.
enum MandelbrotImpl {
    SCALAR,
    VECTOR,
    VECTOR_ILP,
    VECTOR_MULTICORE,
    VECTOR_MULTICORE_MULTITHREAD,
    VECTOR_MULTICORE_MULTITHREAD_ILP,
    ALL
};

// Command-line arguments parser.
int ParseArgsAndMakeSpec(
    int argc,
    char *argv[],
    uint32_t *img_size,
    uint32_t *max_iters,
    MandelbrotImpl *impl) {
    char *implementation_str = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-r") == 0) {
            if (i + 1 < argc) {
                *img_size = atoi(argv[++i]);
                if (*img_size % 32 != 0) {
                    std::cerr << "Error: Image width must be a multiple of 32"
                              << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: No value specified for -r" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                *max_iters = atoi(argv[++i]);
            } else {
                std::cerr << "Error: No value specified for -b" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-i") == 0) {
            if (i + 1 < argc) {
                implementation_str = argv[++i];
                if (strcmp(implementation_str, "scalar") == 0) {
                    *impl = SCALAR;
                } else if (strcmp(implementation_str, "vector") == 0) {
                    *impl = VECTOR;
                } else if (strcmp(implementation_str, "vector_ilp") == 0) {
                    *impl = VECTOR_ILP;
                } else if (strcmp(implementation_str, "vector_multicore") == 0) {
                    *impl = VECTOR_MULTICORE;
                } else if (
                    strcmp(implementation_str, "vector_multicore_multithread") == 0) {
                    *impl = VECTOR_MULTICORE_MULTITHREAD;
                } else if (
                    strcmp(implementation_str, "vector_multicore_multithread_ilp") == 0) {
                    *impl = VECTOR_MULTICORE_MULTITHREAD_ILP;
                } else if (strcmp(implementation_str, "all") == 0) {
                    *impl = ALL;
                } else {
                    std::cerr << "Error: unknown implementation" << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: No value specified for -i" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Unknown flag: " << argv[i] << std::endl;
            return 1;
        }
    }
    std::cout << "Testing with image size " << *img_size << "x" << *img_size << " and "
              << *max_iters << " max iterations." << std::endl;

    return 0;
}

// Output image writers: BMP file header structure
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType{0x4D42};   // File type, always "BM"
    uint32_t fileSize{0};        // Size of the file in bytes
    uint16_t reserved1{0};       // Always 0
    uint16_t reserved2{0};       // Always 0
    uint32_t dataOffset{54};     // Start position of pixel data
    uint32_t headerSize{40};     // Size of this header (40 bytes)
    int32_t width{0};            // Image width in pixels
    int32_t height{0};           // Image height in pixels
    uint16_t planes{1};          // Number of color planes
    uint16_t bitsPerPixel{24};   // Bits per pixel (24 for RGB)
    uint32_t compression{0};     // Compression method (0 for uncompressed)
    uint32_t imageSize{0};       // Size of raw bitmap data
    int32_t xPixelsPerMeter{0};  // Horizontal resolution
    int32_t yPixelsPerMeter{0};  // Vertical resolution
    uint32_t colorsUsed{0};      // Number of colors in the color palette
    uint32_t importantColors{0}; // Number of important colors
};
#pragma pack(pop)

void writeBMP(const char *fname, uint32_t img_size, const std::vector<uint8_t> &pixels) {
    uint32_t width = img_size;
    uint32_t height = img_size;

    BMPHeader header;
    header.width = width;
    header.height = height;
    header.imageSize = width * height * 3;
    header.fileSize = header.dataOffset + header.imageSize;

    std::ofstream file(fname, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    file.write(reinterpret_cast<const char *>(pixels.data()), pixels.size());
}

std::vector<uint8_t> iters_to_colors(
    uint32_t img_size,
    uint32_t max_iters,
    const std::vector<uint32_t> &iters) {
    uint32_t width = img_size;
    uint32_t height = img_size;
    uint32_t min_iters = max_iters;
    for (uint32_t i = 0; i < img_size; i++) {
        for (uint32_t j = 0; j < img_size; j++) {
            min_iters = std::min(min_iters, iters[i * img_size + j]);
        }
    }
    float log_iters_min = log2f(static_cast<float>(min_iters));
    float log_iters_range =
        log2f(static_cast<float>(max_iters) / static_cast<float>(min_iters));
    auto pixel_data = std::vector<uint8_t>(width * height * 3);
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            uint32_t iter = iters[i * width + j];

            uint8_t r = 0, g = 0, b = 0;
            if (iter < max_iters) {
                auto log_iter = log2f(static_cast<float>(iter)) - log_iters_min;
                auto intensity = static_cast<uint8_t>(log_iter * 222 / log_iters_range);
                r = 32;
                g = 32 + intensity;
                b = 32;
            }

            auto index = (i * width + j) * 3;
            pixel_data[index] = b;
            pixel_data[index + 1] = g;
            pixel_data[index + 2] = r;
        }
    }
    return pixel_data;
}

// Benchmarking macros and configuration.
static constexpr size_t kNumOfOuterIterations = 10;
static constexpr size_t kNumOfInnerIterations = 1;
#define BENCHPRESS(func, ...) \
    do { \
        std::cout << std::endl << "Running " << #func << " ...\n"; \
        std::vector<double> times(kNumOfOuterIterations); \
        for (size_t i = 0; i < kNumOfOuterIterations; ++i) { \
            auto start = std::chrono::high_resolution_clock::now(); \
            for (size_t j = 0; j < kNumOfInnerIterations; ++j) { \
                func(__VA_ARGS__); \
            } \
            auto end = std::chrono::high_resolution_clock::now(); \
            times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start) \
                           .count() / \
                kNumOfInnerIterations; \
        } \
        std::sort(times.begin(), times.end()); \
        std::stringstream sstream; \
        sstream << std::fixed << std::setw(6) << std::setprecision(2) \
                << times[0] / 1'000'000; \
        std::cout << "  Runtime: " << sstream.str() << " ms" << std::endl; \
    } while (0)

double difference(
    uint32_t img_size,
    uint32_t max_iters,
    std::vector<uint32_t> &result,
    std::vector<uint32_t> &ref_result) {
    int64_t diff = 0;
    for (uint32_t i = 0; i < img_size; i++) {
        for (uint32_t j = 0; j < img_size; j++) {
            diff +=
                abs(int(result[i * img_size + j]) - int(ref_result[i * img_size + j]));
        }
    }
    return diff / double(img_size * img_size * max_iters);
}

void dump_image(
    const char *fname,
    uint32_t img_size,
    uint32_t max_iters,
    const std::vector<uint32_t> &iters) {
    // Dump result as an image.
    auto pixel_data = iters_to_colors(img_size, max_iters, iters);
    writeBMP(fname, img_size, pixel_data);
}

// Main function.
// Compile with:
//  g++ -march=native -O3 -Wall -Wextra -o mandelbrot mandelbrot_cpu.cc
int main(int argc, char *argv[]) {
    // Get Mandelbrot spec.
    uint32_t img_size = 1024;
    uint32_t max_iters = default_max_iters;
    enum MandelbrotImpl impl = ALL;
    if (ParseArgsAndMakeSpec(argc, argv, &img_size, &max_iters, &impl))
        return -1;

    // Allocate memory.
    std::vector<uint32_t> result(img_size * img_size);
    std::vector<uint32_t> ref_result(img_size * img_size);

    // Compute the reference solution
    mandelbrot_cpu_scalar(img_size, max_iters, ref_result.data());

    // Test the desired kernels.
    if (impl == SCALAR || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_scalar, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_scalar.bmp", img_size, max_iters, result);
    }

#ifdef HAS_VECTOR_IMPL
    if (impl == VECTOR || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector_256, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_vector_256.bmp", img_size, max_iters, result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;

        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector_512, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_vector_512.bmp", img_size, max_iters, result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }
#endif

    if (impl == VECTOR_ILP || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector_ilp, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_vector_ilp.bmp", img_size, max_iters, result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    if (impl == VECTOR_MULTICORE || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector_multicore, img_size, max_iters, result.data());
        dump_image(
            "out/mandelbrot_cpu_vector_multicore.bmp",
            img_size,
            max_iters,
            result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    if (impl == VECTOR_MULTICORE_MULTITHREAD || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(
            mandelbrot_cpu_vector_multicore_multithread,
            img_size,
            max_iters,
            result.data());
        dump_image(
            "out/mandelbrot_cpu_vector_multicore_multithread.bmp",
            img_size,
            max_iters,
            result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    if (impl == VECTOR_MULTICORE_MULTITHREAD_ILP || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(
            mandelbrot_cpu_vector_multicore_multithread_ilp,
            img_size,
            max_iters,
            result.data());
        dump_image(
            "out/mandelbrot_cpu_vector_multicore_multithread_ilp.bmp",
            img_size,
            max_iters,
            result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    return 0;
}
