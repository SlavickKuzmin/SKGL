#include "RenderOnGPU.cuh"


RenderOnGPU::RenderOnGPU(ModelBuffer *model, int width, int height)
{
	this->width = width;
	this->height = height;

	// make model
	this->model = model;
}

RenderOnGPU::~RenderOnGPU()
{
	delete model;
}

__device__ void part(void* pixels, int pinch, int width, int height, ModelBuffer &mb, int first, int last)
{
	for (int i = first; i < last; i++) {
		for (int j = 0; j < 3; j++) {
			Vec3f v0 = mb.vert(mb.face(i, j));
			Vec3f v1 = mb.vert(mb.face(i, (j + 1) % 3));
			int x0 = (v0.x + 1.f)*width / 2.f;
			int y0 = (v0.y + 1.f)*height / 2.f;
			int x1 = (v1.x + 1.f)*width / 2.f;
			int y1 = (v1.y + 1.f)*height / 2.f;
			line(x0, y0, x1, y1, pixels, pinch);
		}
	}
}

__global__ void draw(void* pixels, int pinch, int width, int height, ModelBuffer mb)
{
	//printf("s");
	int idx = blockIdx.x;
	//printf("idx=%d\n", idx);
	//5022
	// 0 => 3000
	// 3000 => 5022
	//printf("size=%d\n", *(mb.nfaces));
	printf(".");
	//for (int i = 0; i < *(mb.nfaces); i++) {
	//	for (int j = 0; j < 3; j++) {
	//		Vec3f v0 = mb.vert(mb.face(i, j));
	//		Vec3f v1 = mb.vert(mb.face(i, (j + 1) % 3));
	//		int x0 = (v0.x + 1.f)*width / 2.f;
	//		int y0 = (v0.y + 1.f)*height / 2.f;
	//		int x1 = (v1.x + 1.f)*width / 2.f;
	//		int y1 = (v1.y + 1.f)*height / 2.f;
	//		line(x0, y0, x1, y1, pixels, pinch);
	//	}
	//}
	if(idx == 0)
	{
		part(pixels,pinch,width,height,mb, 0, 1000);
	}
	else if (idx == 1)
	{
		part(pixels, pinch, width, height, mb, 1000, 2000);
	}
	else if (idx == 2)
	{
		part(pixels, pinch, width, height, mb, 2000, 3000);
	}
	else if (idx == 3)
	{
		part(pixels, pinch, width, height, mb, 3000, 4000);
	}
	else if (idx == 4)
	{
		part(pixels, pinch, width, height, mb, 4000, 5022);
	}
	//printf("e");
}

void RenderOnGPU::refresh(void* pixels, int pinch, int width, int height)
{
	void *gpuPixels;

	int size = height * pinch;
	cudaMalloc((void**)&gpuPixels, size);
	cudaMemcpy(gpuPixels, pixels, size, cudaMemcpyHostToDevice);

	//drawModel<<<1,1>>>(this->shader, mb, ModelView, Projection, Viewport, zBufferGPU, pixels, pinch, width, height);
	
	/*drawModel << <1, 1 >> > (this->shader, mBuf, ModelView, Projection, Viewport, zBufferGPU, pixels, pinch, width, height);
	cudaDeviceSynchronize();*/
	draw<<<5, 1>>> (gpuPixels, pinch, width, height, *model);
	cudaDeviceSynchronize();

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

	//printf(".");

	cudaMemcpy(pixels, gpuPixels, size, cudaMemcpyDeviceToHost);
	cudaFree(gpuPixels);
	//printf("start-");

	//for (int i = 0; i < model->nfaces(); i++) {
	//	for (int j = 0; j < 3; j++) {
	//		//this->shader->vertex(i, j, ModelView, Projection);
	//	}
	//	//triangle(this->shader->varying_tri, *shader, this->renderer, zbuffer, Viewport);
	//}

	//printf("end ");
}