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

// GPU headers

__global__ void kernel();
__host__ void runKernel();

class RenderOnGPU
{
public:
	// copy model to GPU memory and renderer
	RenderOnGPU(Model *model, int width, int height, SDL_Renderer *renderer);
	//clear all allocated memory
	~RenderOnGPU();
	void line(int x0, int y0, int x1, int y1, TGAColor color);
	void drawModel();
	void refresh();
	void printDeviceInfo();
private:
	Model *gModel;
	SDL_Renderer *cpuRenderer;
	SDL_Renderer *gpuRenderer;
	int width;
	int height;
	void cudasafe(int error, char* message, char* file, int line);
};

#endif // !RefreshOnGPU_CUH_