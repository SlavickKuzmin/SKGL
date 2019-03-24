#include "ModelBuffer.cuh"

ModelBuffer::ModelBuffer(Model *model)
{
	int dnverts = model->verts_.size();
	int dnfaces = model->faces_.size();
	int dnfacesElem = model->faces_[0].size();

	cudaMalloc((void**)&nverts, sizeof(int));
	cudaMalloc((void**)&nfaces, sizeof(int));
	cudaMalloc((void**)&nfacesElem, sizeof(int));

	cudaMemcpy(nverts, &dnverts, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nfaces, &dnfaces, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nfacesElem, &dnfacesElem, sizeof(int), cudaMemcpyHostToDevice);
	
	unsigned long verts_bytes = model->verts_.size() * sizeof(Vec3f);
	unsigned long norms_bytes = model->norms_.size() * sizeof(Vec3f);
	unsigned long uv_bytes = model->uv_.size() * sizeof(Vec2f);

	cudaMalloc((void**)&verts_, verts_bytes);
	cudaMalloc((void**)&norms_, norms_bytes);
	cudaMalloc((void**)&uv_, uv_bytes);

	cudaMemcpy(verts_, model->verts_.data(), verts_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(norms_, model->norms_.data(), norms_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(uv_, model->uv_.data(), uv_bytes, cudaMemcpyHostToDevice);

	unsigned long faces_bytes = model->faces_.size()*sizeof(Vec3i*);
	unsigned long faces_elements_bytes = model->faces_[0].size();

	int height = model->faces_.size();
	int width = dnfacesElem;

	// falatten 2d array from CPU to 1d array on GPU
	int size = height * width;
	Vec3i *flatten = (Vec3i*)malloc(size*sizeof(Vec3i));//new Vec3i[size];
	for (int h = 0; h < height; h++) {
		for (int w = 0; w < width; w++)
			flatten[width * h + w] = model->faces_[h][w];
	}

	cudaMalloc((void**)&faces_, size*sizeof(Vec3i));
	cudaMemcpy(faces_, flatten, size * sizeof(Vec3i), cudaMemcpyHostToDevice);
	free(flatten);
}

__device__ int* ModelBuffer::getNVerts()
{
	return this->nverts;
}
__device__ int* ModelBuffer::getNFaces()
{
	return this->nfaces;
}
__device__ int* ModelBuffer::getNFacesElem()
{
	return this->nfacesElem;
}

ModelBuffer::~ModelBuffer()
{
	cudaFree(verts_);
	cudaFree(norms_);
	cudaFree(uv_);
	cudaFree(faces_);
	cudaFree(nverts);
	cudaFree(nfaces);
	cudaFree(nfacesElem);
}

__device__ Vec3f ModelBuffer::normal(int iface, int nthvert)
{
	int inx0 = (faces_[*nfacesElem * iface + nthvert])[2];
	//int idx = faces_[iface][nthvert][2];
	return norms_[inx0].normalize();
}

__device__ Vec3f ModelBuffer::vert(int i)
{
	return verts_[i];
}

__device__ Vec3f ModelBuffer::vert(int iface, int nthvert)
{
	int idx = (faces_[*nfacesElem * iface + nthvert])[0];
	return verts_[idx];
}

__device__ Vec2f ModelBuffer::uv(int iface, int nthvert)
{
	//return uv_[faces_[iface][nthvert][1]];
	int idx = (faces_[*nfacesElem * iface + nthvert])[0];
	return uv_[idx];
}

__device__ int ModelBuffer::face(int i, int idx) {
	return (faces_[*nfacesElem*i+idx])[0];
}

//width * i + j

//2d:
//data[y*w + x]
// x,y, width
// 3d:
//Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]