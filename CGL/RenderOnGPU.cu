#include "RenderOnGPU.cuh"

RenderOnGPU::RenderOnGPU(Model *model, int width, int height)
{
	this->width = width;
	this->height = height;

	ModelBuffer *mb = new ModelBuffer(model);
	// make model
	// TODO possible leak
	cudaMalloc((void**)&(this->model), sizeof(ModelBuffer));
	cudaMemcpy(this->model, mb, sizeof(ModelBuffer), cudaMemcpyHostToDevice);
	//this->model = mb;
	this->m = model;

	zbuffer = new int[width*height];
	for (int i = 0; i < width*height; i++) {
		zbuffer[i] = std::numeric_limits<int>::min();
	}

	//int *zBufferGPU;
	cudaMalloc((void**)&zBufferGPU, width*height * sizeof(int));
	
	threads_size = 5000;
	int* arr = splitByThreads(m->nfaces(), threads_size);

	//int *cArr;
	cudaMalloc((void**)&cArr, sizeof(int)*(threads_size + 1));
	cudaMemcpy(cArr, arr, sizeof(int)*(threads_size + 1), cudaMemcpyHostToDevice);
	free(arr);
}

RenderOnGPU::~RenderOnGPU()
{
	delete model;
	//{ // dump z-buffer (debugging purposes only)
	//	TGAImage zbimage(width, height, TGAImage::GRAYSCALE);
	//	for (int i = 0; i < width; i++) {
	//		for (int j = 0; j < height; j++) {
	//			zbimage.set(i, j, TGAColor(zbuffer[i + j * width], 1, 1));
	//		}
	//	}
	//	zbimage.flip_vertically(); // i want to have the origin at the left bottom corner of the image
	//	zbimage.write_tga_file("D:\\zbuffer.tga");
	//}
	delete[] zbuffer;
	cudaFree(zBufferGPU);
	cudaFree(cArr);
}


__device__ Vec3f m2v(Matrix m) {
	return Vec3f(m[0][0] / m[3][0], m[1][0] / m[3][0], m[2][0] / m[3][0]);
}

__device__ Matrix v2m(Vec3f v) {
	Matrix m;
	m[0][0] = v.x;
	m[1][0] = v.y;
	m[2][0] = v.z;
	m[3][0] = 1.f;
	return m;
}

__device__ Matrix viewport(int x, int y, int w, int h) {
	Matrix m = Matrix::identity();
	m[0][3] = x + w / 2.f;
	m[1][3] = y + h / 2.f;
	m[2][3] = 255 / 2.f; //depth=255

	m[0][0] = w / 2.f;
	m[1][1] = h / 2.f;
	m[2][2] = 255 / 2.f;//depth=255
	return m;
}


__device__ void part(void* pixels, int pinch, int width, int height, ModelBuffer *mb, int first, int last, int *zbuffer, int ra)
{
	//printf("T");
	// new with textures
	//printf("ra=%d\n", ra);
	Vec3f light_dir(0, 0, -0.1*ra);//todo remove it
	Vec3f camera(0, 0, 3);
	Matrix Projection = Matrix::identity();
	Matrix ViewPort = viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
	Projection[3][2] = -1.f / camera.z;

	const int depth = 255;//todo it too
	for (int i = first; i < last; i++) {
		Vec3i screen_coords[3];
		Vec3f world_coords[3];
		for (int j = 0; j < 3; j++) {
			Vec3f v = mb->vert(mb->face(i, j));
			/*screen_coords[j] = Vec3i((v.x + 1.)*width / 2., (v.y + 1.)*height / 2., (v.z + 1.)*depth / 2.);
			world_coords[j] = v;*/
			screen_coords[j] = m2v(ViewPort*Projection*v2m(v));
			world_coords[j] = v;
		}
		Vec3f n = (world_coords[2] - world_coords[0]) ^ (world_coords[1] - world_coords[0]);
		n.normalize();
		float intensity = n * light_dir;
		if (intensity > 0) {
			//intensity *= ra;
			Vec2i uv[3];
			for (int k = 0; k < 3; k++) {
				uv[k] = mb->uv(i, k);
				//printf("x=%d, y=%d\n", uv[k].x, uv[k].y);
			}
			//printf("h=%d, w=%d, ba=%d\n", mb.diffusemap_.get_height(), mb.diffusemap_.get_width(), mb.diffusemap_.get_bytespp());
			triangleWihTex(screen_coords[0], screen_coords[1], screen_coords[2],
				uv[0], uv[1], uv[2], pixels, pinch, intensity, zbuffer, mb);
		}
	}
}

int* splitByThreads(int model, int parts)
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

__global__ void draw(void* pixels, int pinch, int width, int height, ModelBuffer *mb, int threads_size, int *arr, int *zbuffer, int ra)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//printf("size=%d\n", threads_size);
	
	if (idx < threads_size + 1)
	{
		//debugPrint(arr, threads_size + 1);
		//printf("idx=%d\n", idx);
		part(pixels, pinch, width, height, mb, arr[idx], arr[idx + 1], zbuffer, ra);
	}

}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	}
}

void RenderOnGPU::refresh(void* pixels, int pinch, int width, int height)
{
	void *gpuPixels;
	int size = height * pinch;
	cudaMalloc((void**)&gpuPixels, size);
	cudaMemcpy(gpuPixels, pixels, size, cudaMemcpyHostToDevice);

	cudaMemcpy(zBufferGPU, zbuffer, width*height * sizeof(int), cudaMemcpyHostToDevice);

	clock_t begin = clock();

	//// parts is 7, res array size 8
	//int* arr = splitByThreads(5022, 20);
	//debugPrint(arr, 21);
	//printf(".");

	srand(time(0));
	int ra = (rand() % 20)+5;
	draw <<<128, 64 >>> (gpuPixels, pinch, width, height, model, threads_size, cArr, zBufferGPU, ra);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	printf("time: %lf\n", elapsed_secs);

	cudaMemcpy(pixels, gpuPixels, size, cudaMemcpyDeviceToHost);
	cudaFree(gpuPixels);
}