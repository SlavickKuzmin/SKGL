#include "model.h"
#include "SDL.h"
#include "ModelBuffer.cuh"
#include "cuda_runtime_api.h"
#include "Helpers.cuh"
#include <time.h>

class RenderOnGPU
{
public:
	RenderOnGPU(Model *model, int width, int height);
	~RenderOnGPU();
	void refresh(void* pixels, int pinch, int width, int height);
private:
	ModelBuffer *model;
	Model *m;
	int width;
	int height;
	int *zbuffer;
};