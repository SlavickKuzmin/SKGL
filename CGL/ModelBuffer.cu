#include "ModelBuffer.cuh"

// Construct a model buffer from given CPU-stored model.
// Copy all CPU stored variables to GPU memory.
gl::ModelBuffer::ModelBuffer(Model *model)
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
}

// Gets a pointer to vertex number.
__device__ int* gl::ModelBuffer::getNVerts()
{
	return this->nverts;
}

// Gets a pointer to faces number.
__device__ int* gl::ModelBuffer::getNFaces()
{
	return this->nfaces;
}

// Gets a pointer to faces element number.
__device__ int* gl::ModelBuffer::getNFacesElem()
{
	return this->nfacesElem;
}

// Destructor: free all allocated GPU memory.
gl::ModelBuffer::~ModelBuffer()
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
}

// Return a color for given UV vector.
__device__ gl::Color::Device gl::ModelBuffer::diffuse(Vec2f uvf)
{
	Vec2i uv(uvf[0] * diffuse_texture->getWidth(), uvf[1] * diffuse_texture->getHeight());
	return diffuse_texture->get(uv[0], uv[1]);
}

// Return a normal vector.
__device__ Vec3f gl::ModelBuffer::normal(int iface, int nthvert)
{
	int inx0 = (faces_[*nfacesElem * iface + nthvert])[2];
	return norms_[inx0].normalize();
}

// Return a normal vector from UV vector.
__device__ Vec3f gl::ModelBuffer::normal(Vec2f uvf)
{
	Vec2i uv(uvf[0] * this->normal_map_texture->getWidth(), uvf[1] * this->normal_map_texture->getHeight());
	gl::Color::Device c = this->normal_map_texture->get(uv[0], uv[1]);
	Vec3f res;
	for (int i = 0; i < 3; i++)
		res[2 - i] = (float)c[i] / 255.f*2.f - 1.f;
	return res;
}

// Return a vertex by index. 
__device__ Vec3f gl::ModelBuffer::vert(int i)
{
	return verts_[i];
}

// Return a vertex.
__device__ Vec3f gl::ModelBuffer::vert(int iface, int nthvert)
{
	int idx = (faces_[*nfacesElem * iface + nthvert])[0];
	return verts_[idx];
}

// Return a UV vector.
__device__ Vec2f gl::ModelBuffer::uv(int iface, int nthvert)
{
	int idx = (faces_[*nfacesElem * iface + nthvert])[1];
	return uv_[idx];
}

// Return a face.
__device__ int gl::ModelBuffer::face(int i, int idx) {
	return (faces_[*nfacesElem*i+idx])[0];
}

// Some additional (helps for me) info.
//__fmaf_rd:
//x * y + z

//width * i + j

//2d:
//data[y*w + x]
// x,y, width
// 3d:
//Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]