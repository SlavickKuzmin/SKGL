#ifndef  _CUDA_GL_CUH_
#define _CUDA_GL_CUH_

#include "Color.cuh"
#include "cuda_runtime_api.h"
#include "cuda_geometry.cuh"

#include <SDL.h>

//extern Matrix ModelView;
//extern Matrix Projection;

__device__ void viewport(int x, int y, int w, int h, Matrix &Viewport);
__device__ void projection(float coeff, Matrix &Projection); // coeff = -1/c
__device__ void lookat(Vec3f eye, Vec3f center, Vec3f up, Matrix &ModelView);

struct IShader {
	__device__ virtual ~IShader();
	__device__ virtual Vec4f vertex(int iface, int nthvert) = 0;
	__device__ virtual bool fragment(Vec3f bar, Color &color) = 0;
};

__device__ void triangle(mat<4, 3, float> &pts, IShader &shader, void *pixels, int pinch, float *zbuffer, Matrix &Viewport);

#endif //  _CUDA_GL_CUH_
