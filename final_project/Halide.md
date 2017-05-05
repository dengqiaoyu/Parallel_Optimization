## Vectorize

```
Var x_outer, x_inner;
gradient.split(x, x_outer, x_inner, 4);
gradient.vectorize(x_inner);
Buffer<int> output = gradient.realize(8, 4);
```

Or

```c
Var x_outer, x_inner;
gradient.split(x, 4);
Buffer<int> output = gradient.realize(8, 4);
```

## Unrolling a loop

```c
Var x_outer, x_inner;
gradient.split(x, x_outer, x_inner, 2);
gradient.unroll(x_inner);
```

equals to:

```c++
for (int y = 0; y < 4; y++) {
  for (int x_outer = 0; x_outer < 2; x_outer++) {
    // Instead of a for loop over x_inner, we get two
    // copies of the innermost statement.
    {
      int x_inner = 0;
      int x = x_outer * 2 + x_inner;
      printf("Evaluating at x = %d, y = %d: %d\n", x, y, x + y);
    }
    {
      int x_inner = 1;
      int x = x_outer * 2 + x_inner;
      printf("Evaluating at x = %d, y = %d: %d\n", x, y, x + y);
    }
  }
}
```

## Fusing, tiling and paralleling

```c++
Func gradient("gradient_fused_tiles");
gradient(x, y) = x + y;
		// let x, y be divieded by 4x4 Marix, the index of matrix is detemined by
		// x_outer, y_outer, x_inner, y_inner.
gradient.tile(x, y, x_outer, y_outer, x_inner, y_inner, 4, 4);
		// Convert x_outer, y_outer to a single tile_index,
		// x_outer = tile_index % 4
		// y_outer = tile_index / 4
		.fuse(x_outer, y_outer, tile_index)
        // Parallel on all the blocks
        .parallel(tile.index);
Buffer<int> output = gradient.realize(8, 8);
```

## Parallel and vectorize

```c++
// Are you ready? We're going to use all of the features above now.
Func gradient_fast("gradient_fast");
gradient_fast(x, y) = x + y;

// We'll process 64x64 tiles in parallel.
// Divide image into 64x64 block, and parallel on them.
Var x_outer, y_outer, x_inner, y_inner, tile_index;
gradient_fast
  .tile(x, y, x_outer, y_outer, x_inner, y_inner, 64, 64)
  .fuse(x_outer, y_outer, tile_index)
  .parallel(tile_index);


// We'll compute two scanlines at once while we walk across
// each tile. We'll also vectorize in x. The easiest way to
// express this is to recursively tile again within each tile
// into 4x2 subtiles, then vectorize the subtiles across x and
// unroll them across y:
Var x_inner_outer, y_inner_outer, x_vectors, y_pairs;
gradient_fast
  .tile(x_inner, y_inner, x_inner_outer, y_inner_outer, x_vectors, y_pairs, 4, 2)
  // Use SSE or AVX to computer x_vector length data at once
  .vectorize(x_vectors)
  .unroll(y_pairs);

// Note that we didn't do any explicit splitting or
// reordering. Those are the most important primitive
// operations, but mostly they are buried underneath tiling,
// vectorizing, or unrolling calls.

// Now let's evaluate this over a range which is not a
// multiple of the tile size.

// If you like you can turn on tracing, but it's going to
// produce a lot of printfs. Instead we'll compute the answer
// both in C and Halide and see if the answers match.
Buffer<int> result = gradient_fast.realize(350, 250);
```



# Multi-stage pipeline

## computer at x

![compute at x](http://halide-lang.org/tutorials/figures/lesson_08_store_root_compute_x.gif)

```c
// The performance characteristics of this strategy are the
// best so far. One of the four values of the producer we need
// is probably still sitting in a register, so I won't count
// it as a load:
// producer.store_root().compute_at(consumer, x):
// - Temporary memory allocated: 10 floats
// - Loads: 48
// - Stores: 56
// - Calls to sin: 40
```



# Multi-pass Func

## Case 4

```c++
Func producer, consumer;
producer(x, y) = (x * y) / 10 + 8;
consumer(x, y) = x + y;
consumer(x, 0) = producer(x, x);
consumer(0, y) = producer(y, 9-y);
```

Cannot use ``producer.compute_at(consumer, x)`` or ``producer.compute_at(consumer, y)``, because producer is used both for iterating x and y, in order to do that we need create two handlers to do that part.

```c++
Func producer_1, producer_2, consumer_2;
producer_1(x, y) = producer(x, y);
producer_2(x, y) = producer(x, y);

consumer_2(x, y) = x + y;
consumer_2(x, 0) += producer_1(x, x);
consumer_2(0, y) += producer_2(y, 9-y);

// The wrapper functions give us two separate handles on
// the producer, so we can schedule them differently.
producer_1.compute_at(consumer_2, x);
producer_2.compute_at(consumer_2, y);

Buffer<int> halide_result = consumer_2.realize(10, 10);
```

![Multi pass func]()