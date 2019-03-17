#ifndef _CUDA_GL_CUH_
#define _CUDA_GL_CUH_

#include "Color.cuh"
#include "cuda_runtime_api.h"
#include "cuda_geometry.cuh"
#include <SDL.h>
#include "math.h"
#include <limits.h>
#include <cstdlib>
#include <algorithm>
#include "Helpers.cuh"
#include "Shader.cuh"

__device__ void viewport(int x, int y, int w, int h, Matrix &Viewport);
__device__ void projection(float coeff, Matrix &Projection); // coeff = -1/c
__device__ void lookat(Vec3f eye, Vec3f center, Vec3f up, Matrix &ModelView);

__device__ void triangle(mat<4, 3, float> &pts, Shader &shader, void *pixels, int pinch, float *zbuffer, Matrix &Viewport);

#endif //  _CUDA_GL_CUH_
