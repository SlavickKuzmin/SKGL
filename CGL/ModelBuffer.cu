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

	//init diffuse texture
	// TODO possible leak
	TextureBuffer *diffText = new TextureBuffer(model->diffusemap_);
	cudaMalloc((void**)&(this->diffuse_texture), sizeof(TextureBuffer));
	cudaMemcpy(this->diffuse_texture, diffText, sizeof(TextureBuffer), cudaMemcpyHostToDevice);

	// init nomal map texture
	TextureBuffer *normTex = new TextureBuffer(model->normalmap_);
	cudaMalloc((void**)&(this->normal_map_texture), sizeof(TextureBuffer));
	cudaMemcpy(this->normal_map_texture, normTex, sizeof(TextureBuffer), cudaMemcpyHostToDevice);

	printf("Constr call\n");
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
	
	// free texture
	cudaFree(this->diffuse_texture);
	cudaFree(this->normal_map_texture);
	printf("Destr call\n");
}

__device__ Color ModelBuffer::diffuse(Vec2f uvf)
{
	//if (uv.x < 0 || uv.y < 0 || uv.x >= diffuse_texture->getWidth() || uv.y >= diffuse_texture->getHeight()) {
	//	return Color();
	//}
	//Color c(diffuse_texture->texture_binary_data + (uv.x + uv.y*diffuse_texture->getWidth()*(diffuse_texture->getBytesApp()),
	//	diffuse_texture->getBytesApp());
	//c.alpha = 255;
	//return c;
	Vec2i uv(uvf[0] * diffuse_texture->getWidth(), uvf[1] * diffuse_texture->getHeight());
	return diffuse_texture->get(uv[0], uv[1]);
}

__device__ Vec3f ModelBuffer::normal(int iface, int nthvert)
{
	int inx0 = (faces_[*nfacesElem * iface + nthvert])[2];
	return norms_[inx0].normalize();
}
__device__ Vec3f ModelBuffer::normal(Vec2f uvf)
{
	Vec2i uv(uvf[0] * this->normal_map_texture->getWidth(), uvf[1] * this->normal_map_texture->getHeight());
	Color c = this->normal_map_texture->get(uv[0], uv[1]);
	Vec3f res;
	for (int i = 0; i < 3; i++)
		res[2 - i] = (float)c[i] / 255.f*2.f - 1.f;
	return res;
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

//__device__ Vec2i ModelBuffer::uv(int iface, int nthvert)
//{
//	int idx = (faces_[*nfacesElem * iface + nthvert])[1];
//	return Vec2i(uv_[idx].x*(diffuse_texture->getWidth()), uv_[idx].y*(diffuse_texture->getHeight()));
//}

__device__ Vec2f ModelBuffer::uv(int iface, int nthvert)
{
	//int idx = (faces_[*nfacesElem * iface + nthvert])[1];
	//return Vec2i(uv_[idx].x*(diffuse_texture->getWidth()), uv_[idx].y*(diffuse_texture->getHeight()));
	int idx = (faces_[*nfacesElem * iface + nthvert])[1];
	return uv_[idx];
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