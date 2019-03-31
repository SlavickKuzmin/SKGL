#ifndef _MODEL_BUFFER_CUH_
#define _MODEL_BUFFER_CUH_

#include "cuda_geometry.cuh"
#include "cuda_runtime_api.h"
#include "model.h"
#include "Color.cuh"

class ModelBuffer {
public:
	Vec3f *verts_;
	Vec3i *faces_; // 2darray attention, this Vec3i means vertex/uv/normal
	Vec3f *norms_;
    Vec2f *uv_;
	ModelBuffer(Model *model);
	~ModelBuffer();
	int *nverts;
    int *nfaces;
	int *nfacesElem;
	__device__ int* getNVerts();
	__device__ int* getNFaces();
	__device__ int* getNFacesElem();
	__device__ Vec3f normal(int iface, int nthvert);
	__device__ Vec3f vert(int i);
	__device__ Vec3f vert(int iface, int nthvert);
	__device__ Vec2i uv(int iface, int nthvert);
	__device__ int face(int i, int idx);
	__device__ Color diffuse(Vec2i uvf);
	//diff text
	unsigned char *diffuse_data;
	int *diffuse_width;
	int *diffuse_height;
	int *diffuse_bytespp;
};

#endif