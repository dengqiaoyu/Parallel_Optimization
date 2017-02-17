# A Simple CUDA Renderer

* Qiaoyu Deng (qdeng@andrew.cmu.edu)
* Changkai Zhou (zchangka@andrew.cmu.edu)

## Part 1: SAXPY (5 pts)



1) Results on ghc31 (with GTX 1080)

|                    | Time (ms) | Bandwidth (GB/s) |
| ------------------ | :-------: | :--------------: |
| CPU-based          |   21.1    |       14.1       |
| GPU without memcpy |    1.1    |      211.8       |
| GPU with memcpy    |   25.4    |       8.8        |

Low arithmetic intensity causes the major difference of time for GPU between the one without *memcpy* and the one with *memcpy*. PCIe-x16 bus is the bottleneck of the runtime due to relatively high cost from main memory access.

2) Memory bandwidth is 320 GB/s according to [the specification](https://www.nvidia.com/en-us/geforce/products/10series/geforce-gtx-1080/), which is almost consistent with the estimated bandwidth (211.8 GB/s). 

The bandwidth of PCIe-x16 3.0 is 32 GB/s according to [this article](http://www.trentonsystems.com/applications/pci-express-interface/), which is almost consistent with the estimated bandwidth (14.1 GB/s). 


## Part 2: Parallel Prefix-Sum (10 pts)

Results on ghc33 (with GTX 1080):

* Exclusive scan

| Element Count | Fast Time | Your Time    | Score |
| ------------- | --------- | ------------ | ----- |
| 10000         | 0.399     | 0.381        | 1.25  |
| 100000        | 0.782     | 0.593        | 1.25  |
| 1000000       | 1.754     | 1.582        | 1.25  |
| 2000000       | 2.797     | 2.562        | 1.25  |
|               |           | Total score: | 5/5   |

* Find repeat

| Element Count | Fast Time | Your Time    | Score |
| ------------- | --------- | ------------ | ----- |
| 10000         | 0.454     | 0.424        | 1.25  |
| 100000        | 1.070     | 0.803        | 1.25  |
| 1000000       | 2.425     | 1.879        | 1.25  |
| 2000000       | 3.674     | 3.039        | 1.25  |
|               |           | Total score: | 5/5   |


## Part 3: A Simple Circle Renderer

#### Results on [ghc27]

**[Insert your data]**

| Scene Name | Fast Time (Tf) | Your Time (T) | Score |
| ---------- | -------------- | ------------- | ----- |
| rgb        | 0.2999         | 0.2015        | 13    |
| rand10k    | 11.3350        | 14.4000       | 11    |
| rand100k   | 116.8904       | 168.7682      | 10    |
| pattern    | 0.8856         | 0.8274        | 13    |
| snowsingle | 49.8459        | 37.3312       | 13    |
|            |                | Total score:  | 60/65 |


#### Decomposition

We apply both **pixel-level** and **circle-level parallelism** in different steps to optimization the problem solving. The problem is decompose into these steps below:

##### 1. **Partitioning**

Equally partition the image into numerous square boxes with same number of pixels in each box.

##### 2. **Assignment**
Assign each box with the indexes of circles intersecting with the box. Here we use the implementation of **Circle-Level Parallelism**. One circle is treated as a thread and planned to **iterate all boxes that the circle intersects** by updating the circle index array of the boxes. The circle index array in each box has the length of the total number of circles and the value of the ***i***-th index signifies whether the *i*-th circle and the box are intersecting (0-No, 1-Yes). 

**[How you assign work to CUDA thread blocks/threads/even warps]**

##### 3. **Scanning**
Use **exclusive_scan** method to scan each index array in each box and compress them **into minimum space**. 

**[How you assign work to CUDA thread blocks/threads/even warps]**

##### 4. **Rendering**
Render each pixel with the circles that cover the pixel by iterating only the circles in the box where the pixel locates. Here, we use **Pixel-Level Parallelism** and treat each pixel as a thread.

**[How you assign work to CUDA thread blocks/threads/even warps]**


[Core assignment]

#### Synchronization

No explicit synchronization thanks to the independence of workload in every parallel computing process.

However, the implicit synchronization occurs at the ending point of **assignment** and **rendering** with the function `cudaDeviceSynchronize()` to wait for going ahead on CPU until all kernels launched on CPU have finished their tasks.

#### Reducing Communication

* Synchronization

During the step of **assignment**, there are two main options of index storation based on **Circle-Level Parallelism**. In each box, if we store the original indexes of circles (eg. indexes = [3, 6, 210, 10323]), it will trigger the conflicts among parallel threads as circles and cause a high cost of synchronization. Therefore, we use the *num(circles)*-length array of 0/1 signals to store indexes of circles, which sacrifices the space for the time by cutting down the synchronization.

Another simple point is to use **Pixel-Level Parallelism** rather than **Circle(Or Box)-Level Parallelism** during **rendering** step. It definitely saves overhead.

* Main Memory Access

Obviously, partitioning the image into many boxes helps avoid that every pixel goes through the whole list of circles, which is an unaffordable cost as for main memory access.

Besides, in the step of **assignment**, it greatly cuts down the overhead caused by main memory access by using **Circle-Level Parallelism** instead of **Box-Level Parallelism**. If using **Box-Level Parallelsim**, the main memory will be accessed by *num(circles) \* num(boxes)* times. However, using **Circle-Level Parallelism** helps cut it down to *num(circles) \* mean(num(boxes per circle intersecting))* times. 

In addition, during the **scanning** step, we copy the index array from main memory into the block shared memory to save overhead.

[Another one]???

#### Other approaches

##### V1.0 

Most intuitive: directly use pixel-level parallelism by iterating all circles to render. Well, definitely a poor idea.

##### V2.0

**Box-Level Parallelism + Pixel-level Parallelism**

Similar decomposition of problem, but use **Box-Level Parallelism** rather than **Circle-Level Parallelism** during the **assignment** step. As is mentioned above, it costs more overhead because of more main memory access.

##### V2.1

**Box-Level Parallelism + Pixel-level Parallelism + Hierarchical Structure**

Optimized version of V2.0. During the **assignment** step, use hierarchical assignments and iteratively decrease the size of boxes for several times. In theory, it will decrease the total access of main memory. However, the result has no obvious improvement.

##### V3.0

**Circle-Level Parallelism + Pixel-level Parallelism**

The idea introduced above without the delicate optimization for main memory access.

##### V3.1

**Circle-Level Parallelism + Pixel-level Parallelism + More Optimization**

The final idea introduced above.

