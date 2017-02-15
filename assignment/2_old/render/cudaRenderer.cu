#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <thrust/device_vector.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"


#define ROUNDED_DIV(x, y) ((x + y - 1) / y)
#define SIDE_LENGTH 32
#define ROW_THREADS_PER_BLOCK_FIND_CIRCLE 8
#define COLUMN_THREADS_PER_BLOCK_FIND_CIRCLE 8
#define THREADS_PER_BLOCK_FIND_CIRCLE (ROW_THREADS_PER_BLOCK_FIND_CIRCLE * \
                                       COLUMN_THREADS_PER_BLOCK_FIND_CIRCLE)
#define ROW_THREADS_PER_BLOCK_REDENER 16
#define COLUMN_THREADS_PER_BLOCK_REDENER 16

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
////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

// struct CircleIndex {
//     int index;
//     float depth;

//     // bool operator< (const CircleIndex& r) const {
//     //     if (depth < r.depth)
//     //         return true;
//     //     else
//     //         return false;
//     // }
// };

// struct CircleIndexCmp {
//     bool operator() (const CircleIndex& p1, const CircleIndex& p2) const {
//         return p1.depth < p2.depth;
//     }
// };

// set<CircleIndex> CircleIndexVec;
int boxRowNum;
int boxColNum;
int* debug_numCircleInBox;
int* debug_indexCircleInBox;
int debug_numCircles;

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;
    int boxRowNum;
    int boxColNum;

    int imageWidth;
    int imageHeight;
    float* imageData;

    int* numCircleInBox;
    int* indexCircleInBox;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height - imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
//
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i + 1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j + 1] += velocity[index3j + 1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j + 1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi) / NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j + 1] = position[index3i + 1] + y;
        position[index3j + 2] = 0.0f;

        // travel scaled unit length
        velocity[index3j] = cosA / 5.0;
        velocity[index3j + 1] = sinA / 5.0;
        velocity[index3j + 2] = 0.0f;
    }
}

// kernelAdvanceHypnosis
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    float* radius = cuConstRendererParams.radius;

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus
    if (radius[index] > cutOff) {
        radius[index] = 0.02f;
    } else {
        radius[index] += 0.01f;
    }
}


// kernelAdvanceBouncingBalls
//
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() {
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3 + 1];
    float oldPosition = position[index3 + 1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition
        return;
    }

    if (position[index3 + 1] < 0 && oldVelocity < 0.f) { // bounce ball
        velocity[index3 + 1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3 + 1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3 + 1] += velocity[index3 + 1] * dt;

    if (fabsf(velocity[index3 + 1] - oldVelocity) < epsilon
            && oldPosition < 0.0f
            && fabsf(position[index3 + 1] - oldPosition) < epsilon) { // stop ball
        velocity[index3 + 1] = 0.f;
        position[index3 + 1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
            (position.x + radius) < -0.f ||
            (position.x - radius) > 1.f) {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr,
           int pixelX, int pixelY) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist) {
        return;
    }



    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // This conditional is in the inner loop, but it evaluates the
    // same direction for all threads so it's cost is not so
    // bad. Attempting to hoist this conditional is not a required
    // student optimization in Assignment 2
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f - p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*) & (cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * index;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // for all pixels in the bonding box
    for (int pixelY = screenMinY; pixelY < screenMaxY; pixelY++) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX = screenMinX; pixelX < screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            shadePixel(index, pixelCenterNorm, p, imgPtr, 0, 0);
            imgPtr++;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");

    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);
    boxRowNum = ROUNDED_DIV(image->height, SIDE_LENGTH);
    boxColNum = ROUNDED_DIV(image->width, SIDE_LENGTH);
    int* indexCircleInBox;
    cudaMalloc(&indexCircleInBox,
               sizeof(int) * boxRowNum * boxColNum * numCircles);
    cudaMemset(indexCircleInBox, 0,
               sizeof(int) * boxRowNum * boxColNum * numCircles);
    int* numCircleInBox;
    cudaMalloc(&numCircleInBox,
               sizeof(int) * boxRowNum * boxColNum);
    cudaMemset(numCircleInBox, 0,
               sizeof(int) * boxRowNum * boxColNum);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    debug_numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;
    params.boxRowNum = boxRowNum;
    params.boxColNum = boxColNum;
    params.indexCircleInBox = indexCircleInBox;
    debug_indexCircleInBox = indexCircleInBox;
    params.numCircleInBox = numCircleInBox;
    debug_numCircleInBox = numCircleInBox;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake <<< gridDim, blockDim>>>();
    } else {
        kernelClearImage <<< gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
    // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake <<< gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls <<< gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis <<< gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) {
        kernelAdvanceFireWorks <<< gridDim, blockDim>>>();
    }
    cudaDeviceSynchronize();
}

__device__ __inline__ int
deviceCheckIntersect(float px, float py, float maxDist,
                     float box_left, float box_right,
                     float box_top, float box_bottom) {
    // float pixelCenterNormX = invWidth *
    //                          (static_cast<float>(pixelX) + 0.5f);
    // float pixelCenterNormY = invHeight *
    //                          (static_cast<float>(pixelY) + 0.5f);
    // float diffX = px - pixelCenterNormX;
    // float diffY = py - pixelCenterNormY;
    // float pixelDist = diffX * diffX + diffY * diffY;
    // if (pixelDist <= maxDist) {
    //     // printf("pixelDist: %f, maxDist: %f\n", pixelDist, maxDist);
    //     *isIntersect = 1;
    // }
    float cloest_x = px < box_left ? box_left : (px < box_right ? px : box_right);
    float cloest_y = py < box_top ? box_top : (py < box_bottom ? py : box_bottom);

    float diffX = px - cloest_x;
    float diffY = py - cloest_y;

    float cloestDist = diffX * diffX + diffY * diffY;

    if (cloestDist <= maxDist)
        return 1;
    else
        return 0;
}

// __device__ __inline__ void
// insert_in_order(thrust::device_vector<CircleIndex> CircleIndexVec,
//                 CircleIndex circleIndexInstance) {
//     for (int i = 0; i < CircleIndexVec.size(); i++) {
//         CircleIndex it = CircleIndexVec[i];
//         if (it.depth < circleIndexInstance.depth) {
//             CircleIndexVec.insert(CircleIndexVec.begin() + i,
//                                   circleIndexInstance);
//         }
//     }
//     // for (thrust::device_vector<CircleIndex>::iterator it = CircleIndexVec.begin();
//     //         it != CircleIndexVec.end();
//     //         it++) {
//     //     if (it->depth > circleIndexInstance->depth) {
//     //         CircleIndexVec.insert(it - CircleIndexVec.begin(),
//     //                               *circleIndexInstance);
//     //         return;
//     //     }
//     // }
//     CircleIndexVec.push_back(circleIndexInstance);
// }

__global__ void kernelFindIntersects() {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    // printf("x: %d, y: %d\n", x, y);
    // return;
    int numCircles = cuConstRendererParams.numCircles;
    // printf("numCircles: %d\n", numCircles);
    // return;
    int boxIndex = y * cuConstRendererParams.boxRowNum + x;
    int numCircleInBox = 0;
    int* indexCircleInBox = (cuConstRendererParams.indexCircleInBox) + boxIndex * numCircles;
    // int leftUpPixelY = y * SIDE_LENGTH;
    // int leftUpPixelX = x * SIDE_LENGTH;
    // // printf("boxIndex: %d, leftUpPixelX: %d, leftUpPixelY: %d, blockDim.x: %d\n",
    // //        boxIndex, leftUpPixelX, leftUpPixelY, blockDim.x);
    // // return;
    // int rightUpPixelY = leftUpPixelY;
    // int rightUpPixelX = leftUpPixelX + SIDE_LENGTH;
    // int leftDownPixelY = leftUpPixelY + SIDE_LENGTH;
    // int leftDownPixelX = leftUpPixelX;
    // int rightDownPixelY = leftUpPixelY + SIDE_LENGTH;
    // int rightDownPixelX = leftUpPixelX + SIDE_LENGTH;
    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;
    float box_left = ((float)x) * SIDE_LENGTH / width;
    float box_right = ((float)x + 1) * SIDE_LENGTH / width;
    float box_top = ((float)y) * SIDE_LENGTH / height;
    float box_bottom = ((float)y + 1) * SIDE_LENGTH / height;

    for (int circleIndex = 0;
            circleIndex < numCircles;
            circleIndex++) {
        int index3 = 3 * circleIndex;
        float px = cuConstRendererParams.position[index3];
        float py = cuConstRendererParams.position[index3 + 1];
        float rad = cuConstRendererParams.radius[circleIndex];
        float maxDist = rad * rad;

        int is_intersected = 0;
        is_intersected = deviceCheckIntersect(px, py, maxDist,
                                              box_left, box_right,
                                              box_top, box_bottom);
        if (is_intersected != 0) {
            indexCircleInBox[numCircleInBox] = circleIndex;
            numCircleInBox++;
        }

        // screenPixelY = leftUpPixelY;
        // for (screenPixelX = leftUpPixelX;
        //         screenPixelX <= rightUpPixelX;
        //         screenPixelX++) {
        //     deviceCheckIntersect(px, py, maxDist,
        //                          screenPixelX, screenPixelY,
        //                          invWidth, invHeight, &is_intersected);
        //     if (is_intersected == 1) {
        //         indexCircleInBox[numCircleInBox] = circleIndex;
        //         numCircleInBox++;
        //         break;
        //     }
        // }
        // if (is_intersected == 1) {
        //     continue;
        // }

        // screenPixelY = leftDownPixelY;
        // for (screenPixelX = leftDownPixelX;
        //         screenPixelX <= rightDownPixelX;
        //         screenPixelX++) {
        //     deviceCheckIntersect(px, py, maxDist,
        //                          screenPixelX, screenPixelY,
        //                          invWidth, invHeight, &is_intersected);
        //     if (is_intersected == 1) {
        //         indexCircleInBox[numCircleInBox] = circleIndex;
        //         numCircleInBox++;
        //         break;
        //     }
        // }
        // if (is_intersected == 1) {
        //     continue;
        // }

        // screenPixelX = leftUpPixelX;
        // for (screenPixelY = leftUpPixelY;
        //         screenPixelY <= leftDownPixelY;
        //         screenPixelY++) {
        //     deviceCheckIntersect(px, py, maxDist,
        //                          screenPixelX, screenPixelY,
        //                          invWidth, invHeight, &is_intersected);
        //     if (is_intersected == 1) {
        //         indexCircleInBox[numCircleInBox] = circleIndex;
        //         numCircleInBox++;
        //         break;
        //     }
        // }
        // if (is_intersected == 1) {
        //     continue;
        // }

        // screenPixelX = rightUpPixelX;
        // for (screenPixelY = rightUpPixelY;
        //         screenPixelY <= rightDownPixelY;
        //         screenPixelY++) {
        //     deviceCheckIntersect(px, py, maxDist,
        //                          screenPixelX, screenPixelY,
        //                          invWidth, invHeight, &is_intersected);
        //     if (is_intersected == 1) {
        //         indexCircleInBox[numCircleInBox] = circleIndex;
        //         numCircleInBox++;
        //         break;
        //     }
        // }
        // if (is_intersected == 1) {
        //     continue;
        // }
    }

    // printf("numCircleInBox: %d\n", numCircleInBox.);
    cuConstRendererParams.numCircleInBox[boxIndex] = numCircleInBox;
}

__global__ void
kernelRenderPixel() {
    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    int numCircleTotal = cuConstRendererParams.numCircles;

    int pixelX = blockDim.x * blockIdx.x + threadIdx.x;
    int pixelY = blockDim.y * blockIdx.y + threadIdx.y;

    if (pixelX > cuConstRendererParams.imageWidth ||
            pixelY > cuConstRendererParams.imageHeight) {
        return;
    }

    int boxX = pixelX / SIDE_LENGTH;
    int boxY = pixelY / SIDE_LENGTH;
    int boxIdx = boxY * cuConstRendererParams.boxRowNum + boxX;

    int* indexCircleInBox =
        (cuConstRendererParams.indexCircleInBox) + boxIdx * numCircleTotal;
    int numCircleInBox = cuConstRendererParams.numCircleInBox[boxIdx] ;

    for (int i = 0; i < numCircleInBox; i++) {
        int circleIndex = indexCircleInBox[i];
        int index3 = 3 * circleIndex;
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);

        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
        float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                             invHeight * (static_cast<float>(pixelY) + 0.5f));
        shadePixel(circleIndex, pixelCenterNorm, p, imgPtr, pixelX, pixelY);
    }
}

void
print_circle_index_in_box(int start, int end) {
    int* hostNumCircles = new int[sizeof(int) * boxRowNum * boxColNum];
    int* hostIndexCircles = new int[sizeof(int) * boxRowNum * boxColNum * debug_numCircles];
    cudaMemcpy(hostNumCircles, debug_numCircleInBox,
               sizeof(int) * boxRowNum * boxColNum, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostIndexCircles, debug_indexCircleInBox,
               sizeof(int) * boxRowNum * boxColNum * debug_numCircles, cudaMemcpyDeviceToHost);
    for (int y = 0; y < boxRowNum; y++) {
        for (int x = 0; x < boxColNum; x++) {
            int boxIndex = y * boxRowNum + x;
            if (hostNumCircles[boxIndex] > 0) {
                printf("boxY: %d, boxX: %d, %d\n", y, x, hostNumCircles[boxIndex]);
                int *indexArray = hostIndexCircles + boxIndex * debug_numCircles;
                for (int i = 0; i < hostNumCircles[boxIndex]; i++) {
                    printf("%d, ", indexArray[i]);
                }
                printf("\n");
            }
        }
    }
    delete[] hostNumCircles;
    delete[] hostIndexCircles;
}

void
CudaRenderer::render() {
    dim3 threadsPerBlockFind(COLUMN_THREADS_PER_BLOCK_FIND_CIRCLE,
                             ROW_THREADS_PER_BLOCK_FIND_CIRCLE, 1);
    dim3 numBlocksFind(ROUNDED_DIV(boxColNum,
                                   COLUMN_THREADS_PER_BLOCK_FIND_CIRCLE),
                       ROUNDED_DIV(boxRowNum,
                                   ROW_THREADS_PER_BLOCK_FIND_CIRCLE), 1);
    kernelFindIntersects <<< numBlocksFind, threadsPerBlockFind>>>();
    cudaCheckError(cudaDeviceSynchronize());
    // print_circle_index_in_box(0, boxRowNum * boxRowNum - 1);
    dim3 threadsPerBlockRender(COLUMN_THREADS_PER_BLOCK_REDENER,
                               ROW_THREADS_PER_BLOCK_REDENER, 1);
    dim3 numBlocksRender(ROUNDED_DIV(image->width, threadsPerBlockRender.x),
                         ROUNDED_DIV(image->height, threadsPerBlockRender.y),
                         1);
    kernelRenderPixel <<< numBlocksRender, threadsPerBlockRender>>>();

    cudaCheckError(cudaDeviceSynchronize());
}
