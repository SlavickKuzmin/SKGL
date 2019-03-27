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
	for (int i = first; i < last; i++) {
		Vec2i screen_coords[3];
		for (int j = 0; j < 3; j++) {
			Vec3f world_coords = mb.vert(mb.face(i,j));
			screen_coords[j] = Vec2i((world_coords.x + 1.)*width / 2., (world_coords.y + 1.)*height / 2.);
		}
		Color col;
		col.alpha = 255;
		col.red = 255;
		col.green = 10;
		col.blue = 0;
		triangle(screen_coords[0], screen_coords[1], screen_coords[2], pixels, pinch, &col);
		// Linew render
		/*line(screen_coords[0].x, screen_coords[0].y, screen_coords[1].x, screen_coords[1].y, pixels, pinch, &col);
		line(screen_coords[1].x, screen_coords[1].y, screen_coords[2].x, screen_coords[2].y, pixels, pinch, &col);
		line(screen_coords[2].x, screen_coords[2].y, screen_coords[0].x, screen_coords[0].y, pixels, pinch, &col);*/
	}
}

__device__ int* splitByThreads(int model, int parts)
{
	int array_size = parts + 1;
	int* part_array = (int*)malloc(array_size*sizeof(int));
	int partInOneThread = model / parts;
	int lastElementSize = (model - (partInOneThread*parts)) + partInOneThread;

	int counter = -partInOneThread;
	for (int i = 0; i < array_size - 1; i++)
	{
		counter = counter + partInOneThread;
		part_array[i] = counter;
	}
	part_array[array_size - 1] = counter + lastElementSize;

	return part_array;
}

__device__ void debugPrint(int *arr, int size)
{
	for (int i = 0; i < size - 1; i++)
	{
		printf("[%d] s=%d, e=%d ", i, arr[i], arr[i + 1]);
	}
	printf("\n");
}

__global__ void draw(void* pixels, int pinch, int width, int height, ModelBuffer mb, int threads_size)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//printf("size=%d\n", threads_size);
	
	if (idx < threads_size + 1)
	{
		int* arr = splitByThreads(*(mb.nfaces), threads_size);
		//debugPrint(arr, threads_size + 1);
		//printf("idx=%d\n", idx);
		printf(".");
		part(pixels, pinch, width, height, mb, arr[idx], arr[idx + 1]);
		free(arr);
	}

}


#define M 2

void RenderOnGPU::refresh(void* pixels, int pinch, int width, int height)
{
	void *gpuPixels;

	int size = height * pinch;
	cudaMalloc((void**)&gpuPixels, size);
	cudaMemcpy(gpuPixels, pixels, size, cudaMemcpyHostToDevice);

	clock_t begin = clock();

	//// parts is 7, res array size 8
	//int* arr = splitByThreads(5022, 20);
	//debugPrint(arr, 21);

	draw <<<64, 2 >>> (gpuPixels, pinch, width, height, *model, 60);

	cudaDeviceSynchronize();

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	//printf("time: %lf\n", elapsed_secs);


	cudaMemcpy(pixels, gpuPixels, size, cudaMemcpyDeviceToHost);
	cudaFree(gpuPixels);
}