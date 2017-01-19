# Program 1

1. See in the srouce code.

2. produce a graph of speedup compared to the reference sequential implementation as a function of the number of cores used FOR VIEW 1. Is speedup linear in the number of cores used? In your writeup hypothesize why this is (or is not) the case?

   ![prog1_q2](/Users/dengqiaoyu/Documents/Learning/CMU/2_Semester/Parallel_Computing_15-618/repo/assignment/1/prog1_mandelbrot_threads/prog1_q2.png)

   No, it is far and far away from linear as we can see from the blue line in the figure above.

   I think there must be some extra overhead when executing multiple threads such as unbalanced computaion cost for each thread, communication cost or content switch latency. So, it can never be linear in the number of cores.

3. How do your measurements explain the speedup graph you previously created?

   ```
   Round 1:
   Thread 0 run time: 55.341430 ms
   Thread 1 run time: 91.689474 ms
   Thread 2 run time: 106.188526 ms
   Thread 3 run time: 106.812619 ms

   Round 2:
   Thread 0 run time: 55.839582 ms
   Thread 1 run time: 66.282260 ms
   Thread 2 run time: 69.522140 ms
   Thread 3 run time: 71.892980 ms

   Round 3:
   Thread 0 run time: 55.286306 ms
   Thread 1 run time: 55.666321 ms
   Thread 3 run time: 57.575151 ms
   Thread 2 run time: 57.705027 ms

   Round 4:
   Thread 0 run time: 55.280241 ms
   Thread 1 run time: 55.275218 ms
   Thread 3 run time: 55.298766 ms
   Thread 2 run time: 55.907081 ms

   Round 5:
   Thread 0 run time: 55.262136 ms
   Thread 1 run time: 55.276773 ms
   Thread 2 run time: 55.305860 ms
   Thread 3 run time: 55.319156 ms
   [mandelbrot thread]:            [55.369] ms
   Wrote image file mandelbrot-thread.ppm
                                   (3.54x speedup from 4 threads)
   ```

   From the result above, each core completes their task for different duration, so there is some ALU that may not work while some others is still busy, the performance of cores are not fully utilized.

4. In your writeup, describe your approach and report the final 16-thread speedup obtained. Also comment on the difference in scaling behavior from 4 to 8 threads vs 8 to 16 threads.

   At last the final speedup for 16 threads is 8.68X, here is my strategy. Since mandelbrot has "connect" property which means it is continuous everywhere, so we can assume that there is very small difference(The number of white pixel and black pixel) between adjcent columns since it changes continuously. So we can also assume that two adjcent columns have very similar computation cost. From this assumption, we can then assign each column to every core one by one and in that situation all cores can have a very close computaion blance, which means we can better utilize multiple cores when computing mandelbrot. The performance benefit form 4 to 8 per thread is 0.5X per thread, however, The performance benefit form 8 to 16 per thread is 0.33875X per thread, as we can see the performance benefit from adding more threads is decreasing as we add more and more thread, I think it is because there exists some overhead such as commnication or scheduling process will slowdown the performance. What's more, the imbalance load among different thread will also affect the speedup and after we create threads more than the number of threads that CPU can handle simultaneously (In this case we have 8-core CPU with 16 threads' hyper threading), the performance will not increase or even decrease.

# Program 2

1. See in source code.

2. Does the vector utilization increase, decrease or stay the same as VECTOR_WIDTH changes? Why?

   ```
   Vector Width:              2
   Vector Utilization:        80.501238%

   Vector Width:              4
   Vector Utilization:        73.577597%

   Vector Width:              8
   Vector Utilization:        69.923057%

   Vector Width:              16
   Vector Utilization:        68.121626%
   ```

   Vector utilization decreases when VECTOR_WIDTH increases from 2 to 16, because more VECTOR_WIDTH means that there are more data computed at the same time, and not all of data will finish computing process at the same time. More data computed at the same time, the more chance we have that many vectors are actually not being used because they have finished. From math perspective, *vector utilization* is calculated by *utilized vector lanes* divided by *total vector lanes*, from the data above, we can see that while *utilized vector lanes* remain mostly the same when VECTOR_WIDTH increasing from 2 to 16, *total vector lanes* increases because of more loop iterations caused by more imbalance computation cost within vector.

3. See in source code.



# Program 3

## Part 1

1. What is the maximum speedup you expect given what you know about those CPUs:

   4, however the real speedup on machine is only 2.99X.

2. Why might the number you observe be less than this ideal?

   Because it is SIMD computation, so all the data need to be performed through the same instruction, however, there some data that can be calculated(diverged) very quickly after several iterations while some data which is within Mandelbrot set that will continue being calculated untill it reaches maximum iterations. So within the same core, there will be different load on different ALU, and some ALU might be idle while others being busy, this lead to the situation that 4-wide SSE vector cannot perform 400% speed up since it is not fully utilized.

## Part 2

1. What speedup do you observe on view 1? What is the speedup over the version of mandelbrot_ispc that does not partition that computation into tasks?

   1. 4.33X speedup from ISPC-task over serial.
   2. 1.45X speedup from ISPC-task over ISPC-nontask.

2. How did you determine how many tasks to create? Why does the number you chose work best?

   I choose 100 tasks to run, it reaches at 17.87X speedup. To fully utilize all cores of CPU, we need at least assign 16 tasks to the CPU. However, when tasks is more than threads that a CPU can handle simultaneously,  tasks are queued into a list, and thread can choose one from pool when it is idle. Assigning more tasks than the cores will improve performance because more tasks can divide computation cost more evenly and thus reduce imbalances among threads.

3. What are differences between the pthread abstraction (used in Program 1) and the ISPC task abstraction?

   1. According to user guide provided by Intel, every task will be assigned to different threads when running, I think the relationship between tasks and threads is just like the relationship between threads and processes. Thread is the container of tasks, and there might be multiple tasks on the same thread.
   2. Also, since tasks within the same thread will never run concurrently, so these tasks actually do not need synchronization to access under parallel execution. However, each thread need synchronization because they can run concurrently. Thinking about executing 1000 tasks compared with 1000 threads, the synchronization between threads might be very complex because we need ensure data integrity among so many threads, however, with tasks we just need to synchronize those tasks that are assigned to different threads.
   3. What's more, I think enough tasks can reach a better load balance because the work is divided into smaller one, without having to consider the complex synchronization like threads.

# Program 4

1. What is the speedup due to SIMD parallelization? What is the speedup due to multi-core parallelization?

   1. 2.64X from ISPC-nontask over serial
   2. 21.7X from ISPC-task over serial

2. Does your modification improve SIMD speedup? Does it improve multi-core speedup (i.e., the benefit of moving from ISPC without- tasks to ISPC with tasks)? Please explain why.

   Set the values in array to 2.9999f, which means all the elements have the same computation cost.

   1. Yes
   2. No
   3. if the data in the 4-wide vector are relatively close to each other, which means that they have the same computation cost, in that situation the ALU in the same core will seldom stay idle when performing SIMD instruction. So this type of data will utilize the advantage of SIMD most. Since it only improve the performance when executing SIMD program, it does not affect the multiple core performance over SIMD program without tasks since from the computation is well distributed by assigning 64 tasks.

3. What is the reason for the loss in efficiency?

   For every 4 continous elements in array, put one element with very high computation cost(2.998f) with three elements that have very low computaion cost(1.0f for example).

   In that case, the computaion cost for 1.0f will be unimportant for both serial version and ISPC version, so the main cost would be computing 2.998f. Since for each 4 elements there will be a high cost, the total number of  loading 2.998f for serial and ISPC version is equal. In the most ideal case, oonly one ALU in the core will be utilized while other ALU will stay idle since 1.0f is very easy to compute its root, and there will be no speedup in ISPC compared with serial version. However, in fact, for this type of input, the ISPC version would be even worse, because SIMD it own has some overhead when being executed.


# Program 5

1. What speedup from using ISPC with tasks do you observe? Explain the performance of this program. Do you think it can be improved?

   ```
   ghc29.ghc.andrew.cmu.edu(8 core 3.2 GHz Intel Core i7):
   [saxpy serial]:         [22.314] ms     [13.356] GB/s   [1.793] GFLOPS
   [saxpy ispc]:           [21.269] ms     [14.012] GB/s   [1.881] GFLOPS
   [saxpy task ispc]:      [7.668] ms      [38.865] GB/s   [5.216] GFLOPS
                                   (2.77x speedup from use of tasks)
                                   (1.05x speedup from ISPC)
                                   (2.91x speedup from task ISPC)
   ```

   ```
   ghc66.ghc.andrew.cmu.edu(quad-core 2.2 GHz Intel Core i7):
   [saxpy serial]:		[29.178] ms	[10.214] GB/s	[1.371] GFLOPS
   [saxpy ispc]:		[28.997] ms	[10.278] GB/s	[1.379] GFLOPS
   [saxpy task ispc]:	[29.994] ms	[9.936] GB/s	[1.334] GFLOPS
   				(0.97x speedup from use of tasks)
   				(1.01x speedup from ISPC)
   				(0.97x speedup from task ISPC)
   ```

   ​

   It shows that ISPC does not improve the performance over the serial version, however, ISPC with tasks  increase the performance by nearly 3X.

   The reason why it cannot be improved by SIMD is that the original serial version:

   ```c
   for (int i=0; i<N; i++) {
           result[i] = scale * X[i] + Y[i];
       }
   ```

   It has 3 memory refrencing for every element in the array including load x, load y and store to result. It spends most of its time wating on memory even if it can exchange to another thread when waiting for data, because it is bandwidth bounded. So, this program achieves low ALU utilization even it uses SIMD or muti-thread algorithm, the time it saves on computing is far less than the time used in waiting data. I think it can be improved by using ```__m128 _mm_fmadd_ps (__m128 a, __m128 b, __m128 c)``` in the FMA instruction to avoid intermediate data flow.

2. As shown above, every time we calculate ```result[i] = scale * X[i] + Y[i]``` first we need to load both ```X``` and ```Y```, so we use 8 bytes here. However set value to ```result``` cost more than 4 bytes, because we need first read ```result``` from memory to cache, and then change the value of result in the case, and then ```result``` in the cache is writen back to memory, so set value to result actually use 8 bytes in total. So, for every element in the array, we will use 16 bytes(4 * sizeof(float)) to go through the bus.

3. I think we can use store instructions which will not produce intermediate result such as ```_mm_fmadd_ps```  to reduce the bandwidth requirement for every element.




