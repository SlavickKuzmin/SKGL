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
		line(screen_coords[0].x, screen_coords[0].y, screen_coords[1].x, screen_coords[1].y, pixels, pinch, &col);
		line(screen_coords[1].x, screen_coords[1].y, screen_coords[2].x, screen_coords[2].y, pixels, pinch, &col);
		line(screen_coords[2].x, screen_coords[2].y, screen_coords[0].x, screen_coords[0].y, pixels, pinch, &col);

		// Linew render
		/*line(screen_coords[0].x, screen_coords[0].y, screen_coords[1].x, screen_coords[1].y, pixels, pinch, &col);
		line(screen_coords[1].x, screen_coords[1].y, screen_coords[2].x, screen_coords[2].y, pixels, pinch, &col);
		line(screen_coords[2].x, screen_coords[2].y, screen_coords[0].x, screen_coords[0].y, pixels, pinch, &col);*/
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
		part(pixels,pinch,width,height,mb, 0, 250);
	}
	else if (idx == 1)
	{
		part(pixels, pinch, width, height, mb, 250, 400);
	}
	else if (idx == 2)
	{
		part(pixels, pinch, width, height, mb, 400, 750);
	}
	else if (idx == 3)
	{
		part(pixels, pinch, width, height, mb, 750, 1000);
	}
	else if (idx == 4)
	{
		part(pixels, pinch, width, height, mb, 1000, 1250);
	}
	else if (idx == 5)
	{
		part(pixels, pinch, width, height, mb, 1250, 1500);
	}
	else if (idx == 6)
	{
		part(pixels, pinch, width, height, mb, 1500, 1750);
	}
	else if (idx == 7)
	{
		part(pixels, pinch, width, height, mb, 1750, 2000);
	}
	else if (idx == 8)
	{
		part(pixels, pinch, width, height, mb, 2000, 2250);
	}
	else if (idx == 9)
	{
		part(pixels, pinch, width, height, mb, 2250, 2500);
	}
	else if (idx == 10)
	{
		part(pixels, pinch, width, height, mb, 2500, 2750);
	}
	else if (idx == 11)
	{
		part(pixels, pinch, width, height, mb, 2750, 3000);
	}
	else if (idx == 12)
	{
		part(pixels, pinch, width, height, mb, 3000, 3250);
	}
	else if (idx == 13)
	{
		part(pixels, pinch, width, height, mb, 3250, 3500);
	}
	else if (idx == 14)
	{
		part(pixels, pinch, width, height, mb, 3500, 3750);
	}
	else if (idx == 15)
	{
		part(pixels, pinch, width, height, mb, 3750, 4000);
	}
	else if (idx == 16)
	{
		part(pixels, pinch, width, height, mb, 4000, 4250);
	}
	else if (idx == 17)
	{
		part(pixels, pinch, width, height, mb, 4250, 4500);
	}
	else if (idx == 18)
	{
		part(pixels, pinch, width, height, mb, 4500, 4750);
	}
	else if (idx == 19)
	{
		part(pixels, pinch, width, height, mb, 4750, 5022);
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

	draw <<<20, 1 >>> (gpuPixels, pinch, width, height, *model);
	cudaDeviceSynchronize();

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	//printf("time: %lf\n", elapsed_secs);


	cudaMemcpy(pixels, gpuPixels, size, cudaMemcpyDeviceToHost);
	cudaFree(gpuPixels);
}