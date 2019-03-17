#include "RenderOnGPU.cuh"


RenderOnGPU::RenderOnGPU(Model *model, int width, int height)
{
	this->width = width;
	this->height = height;

	// make model
	this->model = model;

	this->renderer = renderer;

	// init z-buffer
	this->zbuffer = new float[width*height];
	for (int i = width * height; i--; this->zbuffer[i] = -std::numeric_limits<float>::max());
}

__device__ void RenderOnGPU::devInit()
{
	this->light_dir = Vec3f(1, 1, 1);
	this->eye = Vec3f(1, 1, 3);
	this->center = Vec3f(0, 0, 0);
	this->up = Vec3f(0, 5, 100);

	// init lookat, viewport, proj matrix and light dir
	// TODO: move in refresh method

	lookat(eye, center, up, ModelView);
	viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4, Viewport);
	projection(-1.f / (eye - center).norm(), Projection);
	light_dir = proj<3>((Projection*ModelView*embed<4>(light_dir, 0.f))).normalize();

}

RenderOnGPU::~RenderOnGPU()
{
	delete[] this->zbuffer;
}

void RenderOnGPU::setShader(Shader *shader)
{
	//cuda memcpy
	this->shader = shader;
}

Model* RenderOnGPU::getModel()
{
	// -- cuda memcpy dev to dev
	return this->model;
}

Vec3f* RenderOnGPU::getLight_dir()
{
	return &(this->light_dir);
}

__global__ void drawModel(Shader *shader, ModelBuffer &mb, Matrix ModelView, Matrix Projection, Matrix Viewport,
	float *zbuffer)
{
	for (int i = 0; i < mb.nfaces; i++) {
		for (int j = 0; j < 3; j++) {
			shader->vertex(i, j, ModelView, Projection, mb.uv(i, j), mb.normal(i, j), mb.vert(i, j));
		}
		//triangle(shader->varying_tri, *shader, zbuffer, Viewport);// pixels and pinch ....
	}
}

void RenderOnGPU::refresh(void* pixels, int pinch, int width, int height)
{
	void *gpuPixels;

	int size = height * pinch;
	cudaMalloc((void**)&gpuPixels, size);
	cudaMemcpy(gpuPixels, pixels, size, cudaMemcpyHostToDevice);

	float *zBufferGPU;
	cudaMalloc((void**)zBufferGPU, width*height*sizeof(float));
	cudaMemcpy(zBufferGPU, zbuffer, width*height * sizeof(float), cudaMemcpyHostToDevice);

	ModelBuffer mBuf(model);
	drawModel<<<1,1>>>(this->shader, mBuf, ModelView, Projection, Viewport, zBufferGPU);
	

	//for (int i = 0; i < model->nfaces(); i++) {
	//	std::vector<int> face = model->face(i);
	//	for (int j = 0; j < 3; j++) {
	//		Vec3f v0 = model->vert(face[j]);
	//		Vec3f v1 = model->vert(face[(j + 1) % 3]);
	//		int x0 = (v0.x + 1.)*width / 2.;
	//		int y0 = (v0.y + 1.)*height / 2.;
	//		int x1 = (v1.x + 1.)*width / 2.;
	//		int y1 = (v1.y + 1.)*height / 2.;
	//		line << <1, 1 >> > (x0, y0, x1, y1, gpuPixels, pinch);
	//		cudaDeviceSynchronize();
	//	}
	//}

	printf(".");

	cudaMemcpy(pixels, gpuPixels, size, cudaMemcpyDeviceToHost);
	cudaFree(gpuPixels);
	cudaFree(zBufferGPU);
	//printf("start-");

	//for (int i = 0; i < model->nfaces(); i++) {
	//	for (int j = 0; j < 3; j++) {
	//		//this->shader->vertex(i, j, ModelView, Projection);
	//	}
	//	//triangle(this->shader->varying_tri, *shader, this->renderer, zbuffer, Viewport);
	//}

	//printf("end ");
}