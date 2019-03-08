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
#else
#define CUDA_CALLABLE_MEMBER
#endif 

// CPU headers
#include "model.h"
#include <SDL.h>
#undef main

// GPU headers

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
private:
	Model *gModel;
	SDL_Renderer *cpuRenderer;
	SDL_Renderer *gpuRenderer;
	int width;
	int height;
};

#endif // !RefreshOnGPU_CUH_