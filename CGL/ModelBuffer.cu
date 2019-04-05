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

	//init textures
	cudaMalloc((void**)&diffuse_width, sizeof(int));
	cudaMalloc((void**)&diffuse_height, sizeof(int));
	cudaMalloc((void**)&diffuse_bytespp, sizeof(int));

	cudaMemcpy(diffuse_width, &(model->diffusemap_.width), sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(diffuse_height, &(model->diffusemap_.height), sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(diffuse_bytespp, &(model->diffusemap_.bytespp), sizeof(int), cudaMemcpyHostToDevice);

	unsigned long nbytes = model->diffusemap_.width * model->diffusemap_.height*model->diffusemap_.bytespp;
	cudaMalloc((void**)&diffuse_data, nbytes);

	cudaMemcpy(diffuse_data, model->diffusemap_.data, nbytes, cudaMemcpyHostToDevice);
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
	//cudaFree(verts_);
	//cudaFree(norms_);
	//cudaFree(uv_);
	//cudaFree(faces_);
	//cudaFree(nverts);
	//cudaFree(nfaces);
	//cudaFree(nfacesElem);
	//
	////free textute
	//cudaFree(diffuse_data);
	//cudaFree(diffuse_width);
	//cudaFree(diffuse_height);
	//cudaFree(diffuse_bytespp);
}

__device__ Color ModelBuffer::diffuse(Vec2i uv)
{
	//int x = uvf.x;
	//int y = uvf.y;
	//printf("x=%d, y=%d\n", uv.x, uv.y);
	if (uv.x < 0 || uv.y < 0 || uv.x >= *diffuse_width || uv.y >= *diffuse_height) {
	//	printf("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO");
		return Color();
	}
	Color c(diffuse_data + (uv.x + uv.y * (*diffuse_width))*(*diffuse_bytespp), *diffuse_bytespp);
	c.alpha = 255;
	//printf("r=%d, g=%d, b=%d, a=%d\n", c.red, c.green, c.blue, c.alpha);
	return c;
}

__device__ Vec3f ModelBuffer::normal(int iface, int nthvert)
{
	int inx0 = (faces_[*nfacesElem * iface + nthvert])[2];
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

__device__ Vec2i ModelBuffer::uv(int iface, int nthvert)
{
	//return uv_[faces_[iface][nthvert][1]];
	//int idx = (faces_[*nfacesElem * iface + nthvert])[0];
	//return uv_[idx];

	//int idx = faces_[iface][nthvert][1];
	int idx = (faces_[*nfacesElem * iface + nthvert])[1];
	//printf("x=%f, y=%f  w=d%d, h=%d\n", uv_[idx].x*(*diffuse_width), uv_[idx].y*(*diffuse_height), (*diffuse_width), (*diffuse_height));
	return Vec2i(uv_[idx].x*(*diffuse_width), uv_[idx].y*(*diffuse_height));
}

__device__ int ModelBuffer::face(int i, int idx) {
	return (faces_[*nfacesElem*i+idx])[0];
}

//__fmaf_rd:
//x * y + z

//width * i + j

//2d:
//data[y*w + x]
// x,y, width
// 3d:
//Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]