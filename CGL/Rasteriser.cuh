#ifndef _RASTERISER_CUH_
#define _RASTERISER_CUH_

#include "cuda_geometry.cuh"
#include <float.h> // for max min values
#include "Color.cuh"

__device__ void viewport(Matrix &Viewport, int x, int y, int w, int h);
__device__ void projection(Matrix &Projection, float coeff); // coeff = -1/c
__device__ void lookat(Matrix &ModelView, Vec3f eye, Vec3f center, Vec3f up);
__device__ Vec3f barycentric(Vec2f A, Vec2f B, Vec2f C, Vec2f P);

struct IShader {
	//virtual __~IShader();
	virtual __device__ Vec4f vertex(int iface, int nthvert) = 0;
	virtual __device__ bool fragment(Vec3f bar, Color &color) = 0;
};

//void triangle(Vec4f *pts, IShader &shader, TGAImage &image, float *zbuffer);

#endif // !_RASTERISER_CUH_
