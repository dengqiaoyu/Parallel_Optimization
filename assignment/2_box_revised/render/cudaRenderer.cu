#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"
#include "cycleTimer.h"


#define ROUNDED_DIV(x, y) (((x) + (y) - 1) / (y))
#define BOX_SIDE_LENGTH 32
#define ROW_NUM_BOX_PER_BLOCK 16
#define COLUMN_NUM_BOX_PER_BLOCK 16
#define THREADS_NUM_BOX_PER_BLOCK (ROW_NUM_BOX_PER_BLOCK * \
        COLUMN_NUM_BOX_PER_BLOCK)
#define NUM_CIRCLE_SHRAED_BOX 1024
#define NUM_INDEX_SHARED_BOX 32
#define ROW_THREADS_PER_BLOCK_RENDER BOX_SIDE_LENGTH
#define COLUMN_THREADS_PER_BLOCK_RENDER BOX_SIDE_LENGTH
#define NUM_THREADS_RENDER (ROW_THREADS_PER_BLOCK_RENDER\
        * COLUMN_THREADS_PER_BLOCK_RENDER)
#define MAX_NUM_CIRCLE_IN_SHARED 1024

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

#define MAX(x, y) ((X) > (y) ? (X) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
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

static int width_g;
static int height_g;
static int numCircles_g;
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
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

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
            shadePixel(index, pixelCenterNorm, p, imgPtr);
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
    numCircles_g = numCircles;
    params.imageWidth = image->width;
    width_g = params.imageWidth;
    params.imageHeight = image->height;
    height_g = params.imageHeight;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

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
deviceCheckBox(float px, float py, float maxDist,
               float box_left, float box_right,
               float box_top, float box_bottom) {
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

__global__ void
kernelBox(int* idxCInBox, int* numCInBox,
          int boxColNumInImage) {
    int numCircles = cuConstRendererParams.numCircles;
    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;
    __shared__ float3 sharedCPos[NUM_CIRCLE_SHRAED_BOX];
    __shared__ float sharedRad[NUM_CIRCLE_SHRAED_BOX];
    __shared__ int sharedCIdxInBox[NUM_INDEX_SHARED_BOX * THREADS_NUM_BOX_PER_BLOCK];
    int gBoxX = blockDim.x * blockIdx.x + threadIdx.x;
    int gBoxY = blockDim.y * blockIdx.y + threadIdx.y;
    int tBoxX = threadIdx.x;
    int tBoxY = threadIdx.y;

    int gid = gBoxY * boxColNumInImage + gBoxX;
    int tid = tBoxY * COLUMN_NUM_BOX_PER_BLOCK + tBoxX;

    int* idxCInBoxBase = idxCInBox + gid * numCircles;
    int idxCInBoxLen = 0;
    int numCinBoxLocal = 0;

    // *(float3*)(&cuConstRendererParams.position[index3]);
    int offsetIdxC = 0;
    while (offsetIdxC < numCircles) {
        __syncthreads();
        for (int i = 0;
                i < (NUM_CIRCLE_SHRAED_BOX / THREADS_NUM_BOX_PER_BLOCK);
                i++) {
            int cIdxShared = 4 * tid + i;
            int cIdx = cIdxShared + offsetIdxC;
            if (cIdx < numCircles) {
                int index3 = 3 * cIdx;
                sharedCPos[cIdxShared] =
                    *(float3*)(&cuConstRendererParams.position[index3]);
                sharedRad[cIdxShared] = cuConstRendererParams.radius[cIdx];
            } else {
                sharedRad[cIdxShared] = 0;
                break;
            }

        }
        // __syncthreads();
        // printf("sharedRad[3]: %f\n", sharedRad[3]);
        __syncthreads();

        int cIdxShared = 0;
        while (cIdxShared < NUM_CIRCLE_SHRAED_BOX && \
                sharedRad[cIdxShared] != 0) {
            float3 pos = sharedCPos[cIdxShared];
            float rad = sharedRad[cIdxShared];
            int idxC = cIdxShared + offsetIdxC;

            float box_left = (((float)gBoxX) * BOX_SIDE_LENGTH + 0.5f) / width;
            float box_right = (((float)gBoxX + 1) * BOX_SIDE_LENGTH + 0.5f) / width;
            float box_top = (((float)gBoxY) * BOX_SIDE_LENGTH + 0.5f) / height;
            float box_bottom = (((float)gBoxY + 1) * BOX_SIDE_LENGTH + 0.5f) / height;

            int isIntersected = 0;
            isIntersected = deviceCheckBox(pos.x, pos.y, rad * rad,
                                           box_left, box_right,
                                           box_top, box_bottom);
            if (isIntersected != 0) {
                numCinBoxLocal++;
                idxCInBoxBase[idxCInBoxLen] = idxC;
                idxCInBoxLen++;
            }
            cIdxShared++;
        }
        numCInBox[gid] = numCinBoxLocal;
        offsetIdxC += NUM_CIRCLE_SHRAED_BOX;
        __syncthreads();
    }
}

__device__ __inline__ void
shadePixelShared(float3 p, float rad, float3 rgb,
                 float2 pixelCenter, SceneName sceneName, float4* pixelPtr) {
    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;
    float maxDist = rad * rad;
    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb_real;
    float alpha;

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb_real = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f - p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
    } else {
        rgb_real = rgb;
        alpha = .5f;
    }
    float oneMinusAlpha = 1.f - alpha;
    float4 existingColor = *pixelPtr;
    float4 newColor;
    newColor.x = alpha * rgb_real.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb_real.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb_real.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;
    *pixelPtr = newColor;
}

__global__ void
kernelRenderPixel(int* idxCInBox, int* numCInBox) {
    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;
    float invWidth = 1.f / width;
    float invHeight = 1.f / height;
    int numCircles = cuConstRendererParams.numCircles;
    SceneName sceneName = cuConstRendererParams.sceneName;
    int offsetPerIter = MIN(NUM_THREADS_RENDER, MAX_NUM_CIRCLE_IN_SHARED);

    int pixelX = blockDim.x * blockIdx.x + threadIdx.x;
    int pixelY = blockDim.y * blockIdx.y + threadIdx.y;
    int iIdx = threadIdx.y;
    int jIdx = threadIdx.x;
    int tid = iIdx * COLUMN_THREADS_PER_BLOCK_RENDER + jIdx;

    // return;
    int boxX = pixelX / BOX_SIDE_LENGTH;
    int boxY = pixelY / BOX_SIDE_LENGTH;
    int boxIdx = boxY * (ROUNDED_DIV(width, BOX_SIDE_LENGTH)) + boxX;
    int* cList = idxCInBox + boxIdx * numCircles;
    int numCircleInThisBox = numCInBox[boxIdx];

    __shared__ float4 sharedImageData[ROW_THREADS_PER_BLOCK_RENDER][COLUMN_THREADS_PER_BLOCK_RENDER];
    if (pixelX < width && pixelY < height) {
        sharedImageData[iIdx][jIdx] =
            *(float4*)(&cuConstRendererParams.imageData[4 * (pixelY * width + pixelX)]);
    }
    __syncthreads();
    __shared__ float3 sharedCPos[MAX_NUM_CIRCLE_IN_SHARED];
    __shared__ float sharedRad[MAX_NUM_CIRCLE_IN_SHARED];
    __shared__ float3 sharedColor[MAX_NUM_CIRCLE_IN_SHARED];

    int offsetCircle = 0;
    int cIdx = 0;

    while (offsetCircle < numCircleInThisBox) {
        if (tid >= offsetPerIter) {
        } else if (tid + offsetCircle < numCircleInThisBox && \
                   (cIdx = cList[tid + offsetCircle]) != -1) {
            int index3 = 3 * cIdx;
            sharedCPos[tid] =
                *(float3*)(&cuConstRendererParams.position[index3]);
            sharedRad[tid] = cuConstRendererParams.radius[cIdx];
            sharedColor[tid] =
                *(float3*)(&cuConstRendererParams.color[index3]);
        } else {
            sharedRad[tid] = 0;
        }
        __syncthreads();

        if (pixelX < width && pixelY < height) {
            int iIdxInShared = 0;
            while (sharedRad[iIdxInShared] != 0 && \
                    iIdxInShared < MAX_NUM_CIRCLE_IN_SHARED) {
                float3 CPosRover = sharedCPos[iIdxInShared];
                float radRover = sharedRad[iIdxInShared];
                float3 colorRover = sharedColor[iIdxInShared];

                float2 pixelCenterNorm =
                    make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                invHeight * (static_cast<float>(pixelY) + 0.5f));
                shadePixelShared(CPosRover, radRover, colorRover,
                                 pixelCenterNorm, sceneName,
                                 &sharedImageData[iIdx][jIdx]);
                iIdxInShared++;
            }
        }
        offsetCircle += offsetPerIter;
        __syncthreads();
    }
    if (pixelX < width && pixelY < height) {
        *(float4*)(&cuConstRendererParams.imageData[4 * (pixelY * width + pixelX)]) =
            sharedImageData[iIdx][jIdx];
    }
}

__global__ void
kernelRenderPixelSimple() {
    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;
    float invWidth = 1.f / width;
    float invHeight = 1.f / height;
    int numCircles = cuConstRendererParams.numCircles;
    SceneName sceneName = cuConstRendererParams.sceneName;
    int offsetPerIter = MIN(NUM_THREADS_RENDER, MAX_NUM_CIRCLE_IN_SHARED);

    int pixelX = blockDim.x * blockIdx.x + threadIdx.x;
    int pixelY = blockDim.y * blockIdx.y + threadIdx.y;
    int iIdx = threadIdx.y;
    int jIdx = threadIdx.x;
    int tid = iIdx * COLUMN_THREADS_PER_BLOCK_RENDER + jIdx;

    // return;
    int boxX = pixelX / BOX_SIDE_LENGTH;
    int boxY = pixelY / BOX_SIDE_LENGTH;
    // int boxIdx = boxY * (ROUNDED_DIV(width, BOX_SIDE_LENGTH)) + boxX;
    int numCircleInThisBox = numCircles;

    __shared__ float4 sharedImageData[ROW_THREADS_PER_BLOCK_RENDER][COLUMN_THREADS_PER_BLOCK_RENDER];
    if (pixelX < width && pixelY < height) {
        sharedImageData[iIdx][jIdx] =
            *(float4*)(&cuConstRendererParams.imageData[4 * (pixelY * width + pixelX)]);
    }
    // __syncthreads();
    __shared__ float3 sharedCPos[MAX_NUM_CIRCLE_IN_SHARED];
    __shared__ float sharedRad[MAX_NUM_CIRCLE_IN_SHARED];
    __shared__ float3 sharedColor[MAX_NUM_CIRCLE_IN_SHARED];

    int offsetCircle = 0;
    int cIdx = 0;

    while (offsetCircle < numCircleInThisBox) {
        if (tid >= offsetPerIter) {
        } else if (tid + offsetCircle < numCircleInThisBox && \
                   (cIdx = tid + offsetCircle) != -1) {
            int index3 = 3 * cIdx;
            sharedCPos[tid] =
                *(float3*)(&cuConstRendererParams.position[index3]);
            sharedRad[tid] = cuConstRendererParams.radius[cIdx];
            sharedColor[tid] =
                *(float3*)(&cuConstRendererParams.color[index3]);
        } else {
            sharedRad[tid] = 0;
        }
        __syncthreads();

        if (pixelX < width && pixelY < height) {
            int iIdxInShared = 0;
            while (sharedRad[iIdxInShared] != 0 && \
                    iIdxInShared < MAX_NUM_CIRCLE_IN_SHARED) {
                float3 CPosRover = sharedCPos[iIdxInShared];
                float radRover = sharedRad[iIdxInShared];
                float3 colorRover = sharedColor[iIdxInShared];

                float2 pixelCenterNorm =
                    make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                invHeight * (static_cast<float>(pixelY) + 0.5f));
                shadePixelShared(CPosRover, radRover, colorRover,
                                 pixelCenterNorm, sceneName,
                                 &sharedImageData[iIdx][jIdx]);
                iIdxInShared++;
            }
        }
        offsetCircle += offsetPerIter;
        __syncthreads();
    }
    if (pixelX < width && pixelY < height) {
        *(float4*)(&cuConstRendererParams.imageData[4 * (pixelY * width + pixelX)]) =
            sharedImageData[iIdx][jIdx];
    }
}

void
printCircleIdxInBox(int* idxCInBox, int* numCInBox,
                    int totalNumBox, int numCircles,
                    int boxRowNumInImage, int boxColNumInImage) {
    int* hostnumCInBox = new int[sizeof(int) * totalNumBox];
    int* hostidxCInBox = new int[sizeof(int) * totalNumBox * numCircles];
    cudaMemcpy(hostnumCInBox, numCInBox,
               sizeof(int) * totalNumBox, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostidxCInBox, idxCInBox,
               sizeof(int) * totalNumBox * numCircles, cudaMemcpyDeviceToHost);
    printf("123\n");
    for (int y = 0; y < boxRowNumInImage; y++) {
        for (int x = 0; x < boxColNumInImage; x++) {
            int boxIndex = y * boxColNumInImage + x;
            if (hostnumCInBox[boxIndex] > 0) {
                printf("boxY: %d, boxX: %d, %d\n", y, x, hostnumCInBox[boxIndex]);
                int *indexArray = hostidxCInBox + boxIndex * numCircles;
                for (int i = 0; i < hostnumCInBox[boxIndex]; i++) {
                    printf("%d, ", indexArray[i]);
                }
                printf("\n");
            }
        }
    }
}

void print_int_device_memory(int* device_memory, int len) {
    int* device_memmory_debug = new int[len];
    cudaCheckError(cudaMemcpy(device_memmory_debug, device_memory, len * sizeof(int),
                              cudaMemcpyDeviceToHost));
    printf("printing array, length: %d\n", len);
    // int start = 51 * 4 + 98;
    // int end = start + 3;
    for (int i = 0; i < len; i++) {
        // if (i >= start && i < end) {
        //     printf("%d ", device_memmory_debug[i]);
        // }
        printf("%d ", device_memmory_debug[i]);
    }
    printf("\n");
    delete[] device_memmory_debug;
}

void
CudaRenderer::render() {
    int width = width_g;
    int height = height_g;
    int numCircles = numCircles_g;

    if (numCircles < 10) {
        dim3 blockDimRender(ROW_THREADS_PER_BLOCK_RENDER,
                            COLUMN_THREADS_PER_BLOCK_RENDER);
        dim3 gridDimRender(ROUNDED_DIV(width, blockDimRender.x), ROUNDED_DIV(height, blockDimRender.y));
        kernelRenderPixelSimple <<< gridDimRender, blockDimRender>>>();
        cudaCheckError(cudaDeviceSynchronize());
    } else {
        double startFindTime = CycleTimer::currentSeconds();
        int boxColNumInImage = ROUNDED_DIV(width, BOX_SIDE_LENGTH);
        int boxRowNumInImage = ROUNDED_DIV(height, BOX_SIDE_LENGTH);
        int totalNumBox = boxColNumInImage * boxRowNumInImage;

        dim3 blockDimBox(COLUMN_NUM_BOX_PER_BLOCK, ROW_NUM_BOX_PER_BLOCK);
        dim3 gridDimBox(ROUNDED_DIV(boxColNumInImage, blockDimBox.x), ROUNDED_DIV(boxRowNumInImage, blockDimBox.y));

        int* idxCInBox;
        cudaMalloc(&idxCInBox, sizeof(int) * totalNumBox * numCircles);
        cudaMemset(idxCInBox, -1, sizeof(int) * totalNumBox * numCircles);
        int* numCInBox;
        cudaMalloc(&numCInBox, sizeof(int) * totalNumBox);
        cudaMemset(numCInBox, 0, sizeof(int) * totalNumBox);
        // print_int_device_memory(numCInBox, totalNumBox);

        kernelBox <<< gridDimBox, blockDimBox>>>(idxCInBox, numCInBox, boxColNumInImage);
        cudaCheckError(cudaDeviceSynchronize());
        double endFindTime = CycleTimer::currentSeconds();
        double findTime = endFindTime - startFindTime;
        printf("findTime: %f\n", findTime * 1000.f);
        // print_int_device_memory(numCInBox, totalNumBox);
        // exit(1);
        // printCircleIdxInBox(idxCInBox, numCInBox, totalNumBox, numCircles,
        //                     boxRowNumInImage, boxColNumInImage);
        double startRenderTime = CycleTimer::currentSeconds();
        dim3 blockDimRender(ROW_THREADS_PER_BLOCK_RENDER,
                            COLUMN_THREADS_PER_BLOCK_RENDER);
        dim3 gridDimRender(ROUNDED_DIV(width, blockDimRender.x), ROUNDED_DIV(height, blockDimRender.y));
        kernelRenderPixel <<< gridDimRender, blockDimRender>>>(idxCInBox, numCInBox);
        cudaCheckError(cudaDeviceSynchronize());
        double endRenderTime = CycleTimer::currentSeconds();
        float renderTime = endRenderTime - startRenderTime;
        printf("render: %f\n", renderTime * 1000.f);
    }
}
