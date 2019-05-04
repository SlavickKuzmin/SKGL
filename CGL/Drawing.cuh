/*
 * gpu - means: located in GPU memory. 
 * cpu - means: located in CPU memory (RAM).
 *
 *
 * On every refresh call load renderer on GPU
 * make all manipulations
 * copy renderer from GPU to CPU and display.
 */
#ifndef _HELPERS_CUH_
#define _HELPERS_CUH_

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
#include "ModelBuffer.cuh"
#include "Camera.cuh"
#include "IShader.cuh"

namespace gl
{
	namespace draw
	{
		using namespace gl::computing;

		__device__ void setPixel(void* pixels, int pinch, int x, int y, Color::Device *color);
		__device__ void line(int x0, int y0, int x1, int y1, void* pixels, int pinch, Color::Device *col);
		__device__ void triangle(Vec2i t0, Vec2i t1, Vec2i t2, void* pixels, int pinch, Color::Device *col);
		__device__ void triangleZBuf(Vec3i t0, Vec3i t1, Vec3i t2, void* pixels, int pinch, Color::Device *col, float *zbuffer);
		__device__ void triangleWihTex(Vec3i t0, Vec3i t1, Vec3i t2, Vec2i uv0, Vec2i uv1, Vec2i uv2, void* pixels,
			int pinch, float intensity, int *zbuffer, ModelBuffer *mb);
		__device__ void triangle_s(mat<4, 3, float> *clipc, IShader *shader, void* pixels, int pinch, float *zbuffer, Matrix &Viewport, int ra);

		void SetPixel(Screen *screen, int x, int y, Color::Host color);
		void SetLine(Screen *screen, int x0, int y0, int x1, int y1, Color::Host color);
		void SetTriangle(Screen *screen, int x0, int y0, int x1, int y1, int x2, int y2, Color::Host color);
		void SetRectangle(Screen *screen, int x0, int y0, int x1, int y1, int x2, int y2, int x3, int y3, Color::Host color);
		void SetCircle(Screen *screen, int x0, int y0, int radius, Color::Host color);
		void SetPolygon(Screen *screen, int* coordPair, int coordsSize, Color::Host color);
		void SetImage(Screen *screen, TGAImage *image);
	}
}
#endif // _HELPERS_CUH_