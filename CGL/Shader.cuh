#ifndef _SHADER_CUH_
#define _SHADER_CUH_

#include "model.h"
#include "cuda_geometry.cuh"
#include "cuda_runtime_api.h"

class Shader {
public:
	mat<2, 3, float> varying_uv;  // triangle uv coordinates, written by the vertex shader, read by the fragment shader
	mat<4, 3, float> varying_tri; // triangle coordinates (clip coordinates), written by VS, read by FS
	mat<3, 3, float> varying_nrm; // normal per vertex to be interpolated by FS
	mat<3, 3, float> ndc_tri;     // triangle in normalized device coordinates

	__device__ Vec4f vertex(int iface, int nthvert, Matrix ModelView, Matrix Projection, Vec2f uv, Vec3f normal, Vec3f vert);
	__device__ bool fragment(Vec3f bar, TGAColor &color, Vec3f normal, TGAColor diffuse);

	Shader(Vec3f *light_dir);

	Vec3f light_dir;
};

#endif