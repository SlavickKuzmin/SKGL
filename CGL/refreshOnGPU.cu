#include "refreshOnGPU.cuh"

void RenderOnGPU::line(int x0, int y0, int x1, int y1, TGAColor color)
{
	// brethenhem alg
	bool steep = false;
	if (std::abs(x0 - x1) < std::abs(y0 - y1)) { // if the line is steep, we transpose the image
		std::swap(x0, y0);
		std::swap(x1, y1);
		steep = true;
	}
	if (x0 > x1) { // make it left-to-right
		std::swap(x0, x1);
		std::swap(y0, y1);
	}

	for (int x = x0; x <= x1; x++) {
		float t = (x - x0) / (float)(x1 - x0);
		int y = y0 * (1. - t) + y1 * t;
		if (steep) {
			// if transposed, de-transpose
			SDL_SetRenderDrawColor(this->cpuRenderer, color.bgra[2], color.bgra[1], color.bgra[0], color.bgra[3]);
			SDL_RenderDrawPoint(this->cpuRenderer, y, x);
		}
		else {
			SDL_SetRenderDrawColor(this->cpuRenderer, color.bgra[2], color.bgra[1], color.bgra[0], color.bgra[3]);
			SDL_RenderDrawPoint(this->cpuRenderer, x, y);
		}
	}
}

RenderOnGPU::RenderOnGPU(Model *model, int width, int height, SDL_Renderer *renderer)
{
	this->gModel = model; // TODO move to GPU memory
	this->cpuRenderer = renderer;
	this->width = width;
	this->height = height;
}

RenderOnGPU::~RenderOnGPU()
{
	// TODO add free methods
}

void RenderOnGPU::refresh()
{
	printf("start-");
	drawModel();
	printf("end\n");
}

void RenderOnGPU::drawModel()
{
	for (int i = 0; i < this->gModel->nfaces(); i++) {
		std::vector<int> face = this->gModel->face(i);
		for (int j = 0; j < 3; j++) {
			Vec3f v0 = this->gModel->vert(face[j]);
			Vec3f v1 = this->gModel->vert(face[(j + 1) % 3]);
			int x0 = (v0.x + 1.)*width / 2.;
			int y0 = (v0.y + 1.)*height / 2.;
			int x1 = (v1.x + 1.)*width / 2.;
			int y1 = (v1.y + 1.)*height / 2.;
			line(x0, y0, x1, y1, TGAColor(255, 0, 0, 255));
		}
	}
}

void RenderOnGPU::cudasafe(int error, char* message, char* file, int line) 
{
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s : %i. In %s line %d\n", message, error, file, line);
		exit(-1);
	}
}

void RenderOnGPU::printDeviceInfo()
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

//=======================================================================================================
__global__ void kernel()
{
	printf("kernel");
}

__host__ void runKernel()
{
	//printf("s-");
	kernel<<<1,10>>>();

	cudaDeviceSynchronize();
	//printf("e\n");
}