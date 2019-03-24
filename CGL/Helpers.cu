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
	pixel += ((800-y) * pinch) + (x * sizeof(Uint32));
	*((Uint32*)pixel) = packColorToUint32(color);//abgr
}

__device__ void swap(int &x, int &y)
{
	int tmp = x;
	x = y;
	y = tmp;
}

__device__ void swapVec2i(Vec2i &x, Vec2i &y)
{
	Vec2i tmp = x;
	x = y;
	y = tmp;
}

__device__ void triangle(Vec2i t0, Vec2i t1, Vec2i t2, void* pixels, int pinch, Color *col) {
	if (t0.y == t1.y && t0.y == t2.y) return; // i dont care about degenerate triangles
	// sort the vertices, t0, t1, t2 lower-to-upper (bubblesort yay!)
	if (t0.y > t1.y) swapVec2i(t0, t1);
	if (t0.y > t2.y) swapVec2i(t0, t2);
	if (t1.y > t2.y) swapVec2i(t1, t2);
	int total_height = t2.y - t0.y;
	for (int i = 0; i < total_height; i++) {
		bool second_half = i > t1.y - t0.y || t1.y == t0.y;
		int segment_height = second_half ? t2.y - t1.y : t1.y - t0.y;
		float alpha = (float)i / total_height;
		float beta = (float)(i - (second_half ? t1.y - t0.y : 0)) / segment_height; // be careful: with above conditions no division by zero here
		Vec2i A = t0 + (t2 - t0)*alpha;
		Vec2i B = second_half ? t1 + (t2 - t1)*beta : t0 + (t1 - t0)*beta;
		if (A.x > B.x) swapVec2i(A, B);
		for (int j = A.x; j <= B.x; j++) {
			//image.set(j, t0.y + i, color); // attention, due to int casts t0.y+i != A.y
			setPixel(pixels, pinch, j, t0.y + i, *col);
		}
	}
}

__device__ void line(int x0, int y0, int x1, int y1, void* pixels, int pinch, Color *col) {
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
	for (int x = x0; x <= x1; x++) {
		float t = (x - x0) / (float)(x1 - x0);
		int y = y0 * (1. - t) + y1 * t;
		if (steep) {
			setPixel(pixels, pinch, y, x, *col); // if transposed, de-transpose
		}
		else {
			setPixel(pixels, pinch, x, y, *col);
		}
	}
}