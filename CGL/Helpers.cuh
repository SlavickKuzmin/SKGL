/*
 * gpu - means: located in GPU memory. 
 * cpu - means: located in CPU memory (RAM).
 *
 *
 * On every refresh call load renderer on GPU
 * make all manipulations
 * copy renderer from GPU to CPU and display.
 */
#ifndef RefreshOnGPU_CUH_
#define RefreshOnGPU_CUH_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define KERNEL_MEMBER __global__
#else
#define CUDA_CALLABLE_MEMBER
#define KERNEL_MEMBER
#endif 

#include <cuda.h>
#include "cuda_runtime_api.h"

// CPU headers
#include "model.h"
#include <SDL.h>
#undef main

#include "Screen.h"
#include "Color.cuh"

// GPU headers

__device__ void setPixel(void* pixels, int pinch, int x, int y, Color color);
__global__ void line(int x0, int y0, int x1, int y1, void* pixels, int pinch);
__host__ void drawModel(void* pixels, int pinch, int width, int height, Model *model);

__host__ void printDeviceInfo();
__host__ void cudasafe(int error, char* message, char* file, int line);

#endif // !RefreshOnGPU_CUH_