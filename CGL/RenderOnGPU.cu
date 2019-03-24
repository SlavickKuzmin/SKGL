#include "RenderOnGPU.cuh"


RenderOnGPU::RenderOnGPU(Model *model, int width, int height)
{
	this->width = width;
	this->height = height;

	ModelBuffer *mb = new ModelBuffer(model);
	// make model
	this->model = mb;
}

RenderOnGPU::~RenderOnGPU()
{
	delete model;
}

__device__ void part(void* pixels, int pinch, int width, int height, ModelBuffer &mb, int first, int last)
{
	Color col;
	col.alpha = 255;
	col.red = 255;
	col.green = 0; 
	col.blue = 0;
	//for (int i = first; i < last; i++) {
	//	for (int j = 0; j < 3; j++) {
	//		Vec3f v0 = mb.vert(mb.face(i, j));
	//		Vec3f v1 = mb.vert(mb.face(i, (j + 1) % 3));
	//		int x0 = (v0.x + 1.f)*width / 2.f;
	//		int y0 = (v0.y + 1.f)*height / 2.f;
	//		int x1 = (v1.x + 1.f)*width / 2.f;
	//		int y1 = (v1.y + 1.f)*height / 2.f;
	//		line(x0, y0, x1, y1, pixels, pinch, &col);
	//	}
	//}
	for (int i = first; i < last; i++) {
		Vec2i screen_coords[3];
		for (int j = 0; j < 3; j++) {
			Vec3f world_coords = mb.vert(mb.face(i,j));
			screen_coords[j] = Vec2i((world_coords.x + 1.)*width / 2., (world_coords.y + 1.)*height / 2.);
		}
		//triangle(screen_coords[0], screen_coords[1], screen_coords[2], pixels, pinch, &col);
		line(screen_coords[0].x, screen_coords[0].y, screen_coords[1].x, screen_coords[1].y, pixels, pinch, &col);
		line(screen_coords[1].x, screen_coords[1].y, screen_coords[2].x, screen_coords[2].y, pixels, pinch, &col);
		line(screen_coords[2].x, screen_coords[2].y, screen_coords[0].x, screen_coords[0].y, pixels, pinch, &col);
	}
}

__global__ void draw(void* pixels, int pinch, int width, int height, ModelBuffer mb)
{
	int idx = blockIdx.x;
	//printf("idx=%d\n", idx);
	//printf("size=%d\n", *(mb.nfaces));
	printf(".");
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

	clock_t begin = clock();

	draw <<<5, 1 >>> (gpuPixels, pinch, width, height, *model);
	cudaDeviceSynchronize();

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	printf("time: %lf\n", elapsed_secs);


	cudaMemcpy(pixels, gpuPixels, size, cudaMemcpyDeviceToHost);
	cudaFree(gpuPixels);
}