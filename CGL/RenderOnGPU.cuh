#ifndef _RENDER_ON_GPU_CUH_
#define _RENDER_ON_GPU_CUH_

#include "model.h"
#include "SDL.h"
#include "ModelBuffer.cuh"
#include "cuda_runtime_api.h"
#include <time.h>
#include "Rasteriser.cuh"
#include "Helpers.cuh"

int* splitByThreads(int model, int parts);

class RenderOnGPU
{
public:
	RenderOnGPU(Model *model, int width, int height);
	~RenderOnGPU();
	void refresh(void* pixels, int pinch, int width, int height);
private:
	ModelBuffer *model;
	float *zBufferGPU;
	int *cArr;
	int threads_size;
	Model *m;
	int width;
	int height;
	float *zbuffer;
};

#endif //_RENDER_ON_GPU_CUH_