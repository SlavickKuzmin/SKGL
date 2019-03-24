#include "Helpers.cuh"

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

	cudasafe(cudaGetDeviceCount(&deviceCount), "GetDeviceCount", (char*)__FILE__, __LINE__);

	printf("Number of CUDA devices %d.\n", deviceCount);

	for (int dev = 0; dev < deviceCount; dev++) {
		cudaDeviceProp deviceProp;

		cudasafe(cudaGetDeviceProperties(&deviceProp, dev), "Get Device Properties", (char*)__FILE__, __LINE__);

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
		printf("Total Global Memory:        %u\n", deviceProp.totalGlobalMem);
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
		printf("Number of muliprocessors:   %zd\n", deviceProp.multiProcessorCount);
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

__device__ void swap(int &x, int &y)
{
	int tmp = x;
	x = y;
	y = tmp;
}

__device__ void line(int x0, int y0, int x1, int y1, void* pixels, int pinch) {
	Color col;
	col.alpha = 255;
	col.red = 255;
	col.green = 0; col.blue = 0;

	bool steep = false;
	if (abs(x0 - x1) < abs(y0 - y1)) { // if the line is steep, we transpose the image
		swap(x0, y0);
		swap(x1, y1);
		steep = true;
	}
	if (x0 > x1) { // make it left-to-right
		swap(x0, x1);
		swap(y0, y1);
	}
	//printf("start\n");
	for (int x = x0; x <= x1; x++) {
		float t = (x - x0) / (float)(x1 - x0);
		int y = y0 * (1. - t) + y1 * t;
		if (steep) {
			//printf("s1 x=%d,y=%d", x, y);
			setPixel(pixels, pinch, y, x, col); // if transposed, de-transpose
			//printf("e1");
		}
		else {
			//printf("s x=%d,y=%d", x, y);
			setPixel(pixels, pinch, x, y, col);
			//printf("e");
		}
	}
	//printf("end\n");
}