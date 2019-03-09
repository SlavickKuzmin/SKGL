#ifndef  _CUDA_GL_CUH_
#define _CUDA_GL_CUH_

#include "Color.cuh"
#include "cuda_runtime_api.h"
#include "cuda_geometry.cuh"

#include <SDL.h>

extern Matrix ModelView;
extern Matrix Projection;

void viewport(int x, int y, int w, int h);
void projection(float coeff = 0.f); // coeff = -1/c
void lookat(Vec3f eye, Vec3f center, Vec3f up);

struct IShader {
	virtual ~IShader();
	virtual Vec4f vertex(int iface, int nthvert) = 0;
	virtual bool fragment(Vec3f bar, Color &color) = 0;
};

 void triangle(mat<4, 3, float> &pts, IShader &shader, void *pixels, int pinch, float *zbuffer);

#endif //  _CUDA_GL_CUH_
