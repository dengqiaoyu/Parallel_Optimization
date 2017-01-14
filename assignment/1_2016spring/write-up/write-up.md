# Program 3

## Part 1

1. What is the maximum speedup you expect given what you know about those CPUs:

   4

2. Why might the number you observe be less than this ideal?

   Because it is SIMD computation, so all the data need to be performed through the same instruction, however, there some data that can be calculated(diverged) very quickly after several iterations while some data which is within Mandelbrot set that will continue being calculated untill it reaches maximum iterations. So within the same core, there will be different load on different ALU, and some ALU might be idle while others being busy, this lead to the situation that 4-wide SSE vector cannot perform 400% speed up since it is not fully utilized.

## Part 2

1. What speedup do you observe on view 1? What is the speedup over the version of mandelbrot_ispc that does not partition that computation into tasks?

   1. 7.40X speedup
   2. 1.94X speedup

2. How did you determine how many tasks to create? Why does the number you chose work best?

   I choose 16 tasks to run, which reaches 16.03X speedup over serial version, 4.20X speedup over ISPC with no task version.  

3. What are differences between the pthread abstraction (used in Program 1) and the ISPC task abstraction?

   1. According to user guide provided by Intel, every task will be assigned to different threads when running, I think the relationship between tasks and threads is just like the relationship between threads and processes. Thread is the container of tasks, and there might be multiple tasks on the same thread.
   2. Also, since tasks within the same thread will never run concurrently, so these tasks actually do not need synchronization to access under parallel execution. However, each thread need synchronization because they can run concurrently. Thinking about executing 1000 tasks compared with 1000 threads, the synchronization between threads might be very complex because we need ensure data integrity among so many threads, however, with tasks we just need to synchronize those tasks that are assigned to different threads.
   3. What's more, I think enough tasks can reach a better load balance because the work is divided into smaller one, without having to consider the complex synchronization like threads.

# Program 4

1. What is the speedup due to SIMD parallelization? What is the speedup due to multi-core parallelization?

   1. 2.64X
   2. 17.03X

2. Does your modification improve SIMD speedup? Does it improve multi-core speedup (i.e., the benefit of moving from ISPC without- tasks to ISPC with tasks)? Please explain why.

   Set the values in array in order, which means that two adjcent elements have very few difference.

   1. Yes
   2. No
   3. if the data in the 4-wide vector are relatively close to each other, which means that they have the same computation cost, in that situation the ALU in the same core will seldom stay idle when performing SIMD instruction. So this type of data will utilize the advantage of SIMD most. Since it only improve the performance when executing SIMD program, it does not improve the multiple core performance over SIMD program without tasks.

3. What is the reason for the loss in efficiency?

   For every 4 continous elements in array, put one element with very high computation cost(2.998f) with three elements that have very low computaion cost(1.0f for example).

   In that case, the computaion cost for 1.0f will be unimportant for both serial version and ISPC version, so the main cost would be computing 2.998f. Since for each 4 elements there will be a high cost, the total number of  loading 2.998f for serial and ISPC version is equal. In the most ideal case, oonly one ALU in the core will be utilized while other ALU will stay idle since 1.0f is very easy to compute its root, and there will be no speedup in ISPC compared with serial version. However, in fact, for this type of input, the ISPC version would be even worse, because SIMD it own has some overhead when being executed.

4. ​





