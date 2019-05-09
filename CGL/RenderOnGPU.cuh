#ifndef _RENDER_ON_GPU_CUH_
#define _RENDER_ON_GPU_CUH_

#include "model.h"
#include "SDL.h"
#include "ModelBuffer.cuh"
#include "cuda_runtime_api.h"
#include <time.h>
#include "Camera.cuh"
#include "Drawing.cuh"
#include "IShader.cuh"

using namespace gl::computing;

namespace gl
{
	int* splitByThreads(int model, int parts);

	// Enum for choose render mode.
	enum RenderMode
	{
		Shaders = 0,
		Wire = 1,
		Filled = 2,
		ShadersWithWire = 3
	};

	class RenderOnGPU
	{
	public:
		RenderOnGPU(Model *model, Screen *screen);
		~RenderOnGPU();
		void refresh(Vec3f light_dir, Vec3f eye, Vec3f center, Vec3f up, RenderMode mode);
		float& GetRenderFrameTime();
	private:
		gl::ModelBuffer *model;
		float *zBufferGPU;
		int *cArr;
		int threads_size;
		Model *m;
		Screen *screen;
		int width;
		int height;
		float *zbuffer;
		float renderFrameTime;
		// Screen
		void *h_pixels;
		void *d_pixels;
		int pinch;
	};
}
#endif //_RENDER_ON_GPU_CUH_