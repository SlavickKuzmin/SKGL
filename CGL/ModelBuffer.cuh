#ifndef _MODEL_BUFFER_CUH_
#define _MODEL_BUFFER_CUH_

#include <vector>
#include <string>
#include "cuda_geometry.cuh"
#include "cuda_runtime_api.h"
#include "tgaimage.h"
#include "model.h"

class ModelBuffer {
public:
	Vec3f *verts_;
	Vec3i *faces_; // 2darray attention, this Vec3i means vertex/uv/normal
	Vec3f *norms_;
    Vec2f *uv_;
	ModelBuffer(Model *model);
	~ModelBuffer();
	int nverts;
	int nfaces;
	int nfacesElem;
	__device__ Vec3f normal(int iface, int nthvert);
	__device__ Vec3f vert(int i);
	__device__ Vec3f vert(int iface, int nthvert);
	__device__ Vec2f uv(int iface, int nthvert);
};

#endif