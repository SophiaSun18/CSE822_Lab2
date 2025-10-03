# CSE 822 Lab 2 Write-up

## Compile with the following command on `unicorn`:
g++ mandelbrot_cpu_2.cpp -o cpu_output -mavx2
nvcc mandelbrot_gpu_2.cu -o gpu_output -arch=sm_89

## Question 1
What speedup do you see from increasing the amount of ILP in your CPU and GPU Mandelbrot implementations? What strategy did you use for partitioning the image into groups of vectors, and why did you choose it? How did you deal with managing control flow and state on the CPU and GPU? How many different vectors did you choose to process at once in your CPU and GPU implementations, and why? What seem to be the limiting factors on how far you can scale ILP?

Testing with image size 1024x1024 and 2000 max iterations.

Running mandelbrot_cpu_scalar ...
  Runtime: 663.91 ms

Running mandelbrot_cpu_vector_ilp ...
  Runtime: 295.10 ms
  Correctness: average output difference from reference = 0

Testing with image size 1024x1024 and 2000 max iterations.

Running launch_mandelbrot_gpu_vector_ilp ...
  Runtime: 169.78 ms
  Correctness: average output difference from reference 0

CPU: more than 50% of speedup. GPU: almost 50% of speedup than ILP CPU.
I chose to partition the image along each row, processing 2 vectors of pixels on the same row at once. This approach has better cache locality so the hardware prefetching could easily benefit the program. For both CPU and GPU, I set up separate intermediate variables (x2, y2, w) for each vector and check for both vector's masks in each loop to make sure the inner loop would only terminate when both vectors have completed all computation. I chose to process 2 vectors at once and a key limiting factor is the cache size of the system.

## Question 2
What speedup over the single-core vector-parallel CPU implementation do you see from parallelizing over 8 cores? How do you think the work partitioning strategy might affect the end-to-end performance of the program?

Testing with image size 1024x1024 and 2000 max iterations.

Running mandelbrot_cpu_scalar ...
  Runtime: 663.93 ms

Running mandelbrot_cpu_vector_ilp ...
  Runtime: 294.88 ms
  Correctness: average output difference from reference = 0

NUM_THREAD = 1
Running mandelbrot_cpu_vector_multicore ...
  Runtime: 342.29 ms
  Correctness: average output difference from reference = 0

NUM_THREAD = 8
Running mandelbrot_cpu_vector_multicore ...
  Runtime:  53.78 ms
  Correctness: average output difference from reference = 0

Speedup of 8 cores over 1 core when spliting horizontally: 342.29 / 53.78 = 6.36

Currently the program partitions the image horizontally, assigning each thread with multiple strided rows. Another approach I tried is to divide the image into consecutive chunks of rows and assign each thread a chunk, which might be imbalanced as the last thread might get less workload than other threads and wait idlely for other threads to finish. Although the strided memory access pattern is not consecutive, the access pattern is still predictable so it can both benefit from prefetching and the workload accross each thread is balanced.

## Question 3
What speedup over the single-warp vector-parallel GPU implementation do you see from parallelizing over 192 warp schedulers? How does the absolute run time of the GPU multi-core version compare to the CPU multi-core version? How did you approach designing the work partitioning strategy?

Running mandelbrot_cpu_vector_multicore ...
  Runtime:  53.78 ms
  Correctness: average output difference from reference = 0

mandelbrot_gpu_vector_multicore<<<1, 1>>>(img_size, max_iters, out);
Running launch_mandelbrot_gpu_vector_multicore ...
  Runtime: 2455.72 ms
  Correctness: average output difference from reference 0

mandelbrot_gpu_vector_multicore<<<48, 4 * 32>>>(img_size, max_iters, out);
Running launch_mandelbrot_gpu_vector_multicore ...
  Runtime:   1.15 ms
  Correctness: average output difference from reference 0

Speedup: 2455.72 / 1.15 = 2135.4

The partition strategy: each block takes a strided row, and each thread within the block takes a strided column, similiar to the approach in CPU but added another layer of parallelism.

## Question 4
Try adapting the kernel to use a launch configuration of <<<96, 2 * 32>>>. Will this still assign exactly one warp to every warp scheduler on the machine? How about <<<24, 8 * 32>>>? How do the run times of all of these configurations compare to each other?

mandelbrot_gpu_vector_multicore<<<96, 2 * 32>>>(img_size, max_iters, out);
Running launch_mandelbrot_gpu_vector_multicore ...
  Runtime:   1.11 ms
  Correctness: average output difference from reference 0
Yes. This assigns 2 blocks to each SM, then each block take 2 wrap schedulers in the SM, so in total each warp scheduler gets 1 wrap from 1 of the 2 blocks on the SM. This is slightly faster than the original configuration as it has more blocks so SM can schedule them more flexibly.

mandelbrot_gpu_vector_multicore<<<24, 8 * 32>>>(img_size, max_iters, out);
Running launch_mandelbrot_gpu_vector_multicore ...
  Runtime:   1.40 ms
  Correctness: average output difference from reference 0
No. This assigns 1 block to 1 SM, and only 1/2 of all SMs are assigned tasks. Then within each SM, each wrap scheduler has to take 2 wraps as there're only 4 wrap schedulers but 8 wraps for this SM. This is slightly slower than the original configuration as it only utilizes half of the SMs.

## Question 5
How much are you able to speed up your implementation by introducing multi-threading per-core? What seems to be the optimal number of threads to spawn? What factors do you think might contribute to determining the optimal number of threads?

Testing with image size 1024x1024 and 2000 max iterations.

Running mandelbrot_cpu_scalar ...
  Runtime: 663.63 ms

Running mandelbrot_cpu_vector_ilp ...
  Runtime: 294.93 ms
  Correctness: average output difference from reference = 0

Running mandelbrot_cpu_vector_multicore ...
  Runtime:  59.25 ms
  Correctness: average output difference from reference = 0

Running mandelbrot_cpu_vector_multicore_multithread ...
  Runtime:  22.90 ms
  Correctness: average output difference from reference = 0

The optimal number seems to be 4 threads per core using all 24 cores. I think important factors include the available cores and threads in the hardware, and the available memory bandwidth to execute multiple threads together.

## Question 6
In the `mandelbrot_cpu_vector_multicore_multithread_single_sm` kernel, how does run time vary as a function of the number of warps, beyond the point where there is one warp to populate each of the 4 warp schedulers on the SM? Does it keep improving all the way up to the hard limit of 32 warps per block? If so, by how much? What factors do you think might be contributing to what you observe?

mandelbrot_gpu_vector_multicore_multithread_single_sm<<<1, 4 * 32>>>(img_size, max_iters, out);
Running launch_mandelbrot_gpu_vector_multicore_multithread_single_sm ...
  Runtime:  39.15 ms
  Correctness: average output difference from reference 0

mandelbrot_gpu_vector_multicore_multithread_single_sm<<<1, 8 * 32>>>(img_size, max_iters, out);
Running launch_mandelbrot_gpu_vector_multicore_multithread_single_sm ...
  Runtime:  22.19 ms
  Correctness: average output difference from reference 0

mandelbrot_gpu_vector_multicore_multithread_single_sm<<<1, 16 * 32>>>(img_size, max_iters, out);
Running launch_mandelbrot_gpu_vector_multicore_multithread_single_sm ...
  Runtime:  19.08 ms
  Correctness: average output difference from reference 0

mandelbrot_gpu_vector_multicore_multithread_single_sm<<<1, 24 * 32>>>(img_size, max_iters, out);
Running launch_mandelbrot_gpu_vector_multicore_multithread_single_sm ...
  Runtime:  16.43 ms
  Correctness: average output difference from reference 0

mandelbrot_gpu_vector_multicore_multithread_single_sm<<<1, 32 * 32>>>(img_size, max_iters, out);
Running launch_mandelbrot_gpu_vector_multicore_multithread_single_sm ...
  Runtime:  15.25 ms
  Correctness: average output difference from reference 0

The performance does seem to keep improving all the way up to 32 warps per block, but the speed of improvement gets slower and slower when the number of warps grow larger. When switch from 4 to 8, the speedup is almost 50%, but later the speedup reduces to around 15%, then 10%. Factors such as memory bandwidth and parallelism overhead could be the reason that increasing the number of warps doesn't consistently improves the performance.

## Question 7
As in the CPU case: How much are you able to speed up your GPU implementation by introducing multi-threading per-warp-scheduler? What seems to be the optimal number of warps to spawn? What factors do you think might contribute to determining the optimal number of warps?

mandelbrot_gpu_vector_multicore_multithread_full<<<8, 32 * 32>>>(img_size, max_iters, out);
Running launch_mandelbrot_gpu_vector_multicore_multithread_full ...
  Runtime:   1.81 ms
  Correctness: average output difference from reference 0

mandelbrot_gpu_vector_multicore_multithread_full<<<16, 32 * 32>>>(img_size, max_iters, out);
Running launch_mandelbrot_gpu_vector_multicore_multithread_full ...
  Runtime:   0.92 ms
  Correctness: average output difference from reference 0

mandelbrot_gpu_vector_multicore_multithread_full<<<32, 32 * 32>>>(img_size, max_iters, out);
  Running launch_mandelbrot_gpu_vector_multicore_multithread_full ...
  Runtime:   0.92 ms
  Correctness: average output difference from reference 0

mandelbrot_gpu_vector_multicore_multithread_full<<<48, 32 * 32>>>(img_size, max_iters, out);
  Running launch_mandelbrot_gpu_vector_multicore_multithread_full ...
  Runtime:   0.90 ms
  Correctness: average output difference from reference 0

Using 16 blocks and 32 warps gives the best performance so far for N=1024. After 16 blocks, further increasing the number of blocks only gives tiny improvement in performance, which doesn't match with the extra input of hardware resources. Factors such as the matrix size, total memory bandwidth, overhead of launching blocks and warps, and computation bound could contribute to determining the optimal number of warps.

## Question 8
In your CPU and GPU implementations, how much speedup, if any, were you able to achieve by adding your ILP optimizations back in on top of your multi-core, multi-threaded algorithms? How does the speedup from increasing ILP in this setting compare to the speedup from increasing ILP in the single-threaded, single-core setting? What seems to be the optimal number of threads in this setting on CPU and GPU, and what is the optimal number of independent vectors of pixels to process at once in the inner loop? What factors do you think might be contributing to what you observe?

Testing with image size 1024x1024 and 2000 max iterations.

Running mandelbrot_cpu_scalar ...
  Runtime: 663.88 ms

Running mandelbrot_cpu_vector_ilp ...
  Runtime: 294.93 ms
  Correctness: average output difference from reference = 0

Running mandelbrot_cpu_vector_multicore ...
  Runtime:  67.72 ms
  Correctness: average output difference from reference = 0

Running mandelbrot_cpu_vector_multicore_multithread ...
  Runtime:  23.23 ms
  Correctness: average output difference from reference = 0

Running mandelbrot_cpu_vector_multicore_multithread_ilp ...
  Runtime:  23.33 ms

--------

Testing with image size 1024x1024 and 2000 max iterations.

Running launch_mandelbrot_gpu_vector_ilp ...
  Runtime: 162.68 ms
  Correctness: average output difference from reference 0

Running launch_mandelbrot_gpu_vector_multicore ...
  Runtime:   1.15 ms
  Correctness: average output difference from reference 0

Running launch_mandelbrot_gpu_vector_multicore_multithread_single_sm ...
  Runtime:  15.25 ms
  Correctness: average output difference from reference 0

Running launch_mandelbrot_gpu_vector_multicore_multithread_full ...
  Runtime:   0.92 ms
  Correctness: average output difference from reference 0

Running launch_mandelbrot_gpu_vector_multicore_multithread_full_ilp ...
  Runtime:   1.19 ms
  Correctness: average output difference from reference 0

In both CPU and GPU cases, I don't see any speedup after adding ILP back into the multicore and multithread program. Instead, ILP slows the program down when the program is no longer running in single-core, single-thread. Adding more layers to ILP further decreases the performance. I think it's because earlier steps already configured a good balance for both CPU and GPU programs on the number of cores and threads to use, and the program is already running in good parallel. Adding more workload to each thread doesn't provide any further benefit, but increases the memory bandwidth and syncronization overhead over all the threads.