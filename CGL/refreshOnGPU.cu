#include "refreshOnGPU.cuh"

void cudasafe(int error, char* message, char* file, int line) 
{
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s : %i. In %s line %d\n", message, error, file, line);
		exit(-1);
	}
}

void printDeviceInfo()
{
	int deviceCount;

	cudasafe(cudaGetDeviceCount(&deviceCount), "GetDeviceCount", __FILE__, __LINE__);

	printf("Number of CUDA devices %d.\n", deviceCount);

	for (int dev = 0; dev < deviceCount; dev++) {
		cudaDeviceProp deviceProp;

		cudasafe(cudaGetDeviceProperties(&deviceProp, dev), "Get Device Properties", __FILE__, __LINE__);

		if (dev == 0) {
			if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
				printf("No CUDA GPU has been detected\n");
				return;
			}
			else if (deviceCount == 1) {
				printf("There is 1 device supporting CUDA\n");
			}
			else {
				printf("There are %d devices supporting CUDA\n", deviceCount);
			}
		}

		printf("For device #%d\n", dev);
		printf("Device name:                %s\n", deviceProp.name);
		printf("Major revision number:      %d\n", deviceProp.major);
		printf("Minor revision Number:      %d\n", deviceProp.minor);
		printf("Total Global Memory:        %zu\n", deviceProp.totalGlobalMem);
		printf("Total shared mem per block: %d\n", deviceProp.sharedMemPerBlock);
		printf("Total const mem size:       %d\n", deviceProp.totalConstMem);
		printf("Warp size:                  %d\n", deviceProp.warpSize);
		printf("Maximum block dimensions:   %d x %d x %d\n", deviceProp.maxThreadsDim[0], \
			deviceProp.maxThreadsDim[1], \
			deviceProp.maxThreadsDim[2]);

		printf("Maximum grid dimensions:    %d x %d x %d\n", deviceProp.maxGridSize[0], \
			deviceProp.maxGridSize[1], \
			deviceProp.maxGridSize[2]);
		printf("Clock Rate:                 %d\n", deviceProp.clockRate);
		printf("Number of muliprocessors:   %d\n", deviceProp.multiProcessorCount);
		printf("\nPress any key to continue...\n");
		getchar();
	}
}

__device__ void setPixel(void* pixels, int pinch, int x, int y, Color color)
{
	Uint8 *pixel = (Uint8*)pixels;
	pixel += (y * pinch) + (x * sizeof(Uint32));
	*((Uint32*)pixel) = packColorToUint32(color);//abgr
}

__device__ void line(int x0, int y0, int x1, int y1, void* pixels, int pinch, Color color) {
	int step;
	float dx, dy;
	dx = abs(x1 - x0);
	dy = abs(y1 - y0);

	if (dx >= dy)
		step = dx;
	else
		step = dy;

	dx = dx / (float) step;
	dy = dy / (float) step;

	float x = x0;
	float y = y0;

	int i = 1;
	while (i <= step)
	{
		setPixel(pixels, pinch, x, y, color);
		x = x + dx;
		y = y + dy;
		i = i + 1;
	}
}

__global__ void kernel(void* pixels, int pinch, int width, int height)
{
	Color color;
	color.alpha = 255;
	color.red = 0;
	color.blue = 0;
	color.green = 255;
	line(10, 10, 400, 600, pixels, pinch, color);
	line(800, 200, 0, 20, pixels, pinch, color);
}

__host__ void runKernel(void* pixels, int pinch, int width, int height)
{
	//printf("s-");
	void *gpuPixels;
	int size = height * pinch;
	cudaMalloc((void**)&gpuPixels, size);
	cudaMemcpy(gpuPixels, pixels, size, cudaMemcpyHostToDevice);

	kernel<<<1,1>>>(gpuPixels, pinch, width, height);

	cudaMemcpy(pixels, gpuPixels, size, cudaMemcpyDeviceToHost);
	cudaFree(gpuPixels);

	cudaDeviceSynchronize();
	//printf("e\n");
}