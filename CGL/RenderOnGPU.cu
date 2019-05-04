#include "RenderOnGPU.cuh"

gl::RenderOnGPU::RenderOnGPU(Model *model, Screen *screen)
{
	this->screen = screen;

	this->width = screen->GetWidth();
	this->height = screen->GetHeight();

	ModelBuffer *mb = new ModelBuffer(model);
	// make model
	// TODO possible leak
	cudaMalloc((void**)&(this->model), sizeof(ModelBuffer));
	cudaMemcpy(this->model, mb, sizeof(ModelBuffer), cudaMemcpyHostToDevice);
	//this->model = mb;
	this->m = model;

	zbuffer = new float[width*height];
	for (int i = width * height; i--; zbuffer[i] = -std::numeric_limits<float>::max());

	//int *zBufferGPU;
	cudaMalloc((void**)&zBufferGPU, width*height * sizeof(float));
	
	threads_size = 5022; // diablo_pose
	//threads_size = 2492; // african_head
	int* arr = splitByThreads(m->nfaces(), threads_size);

	//int *cArr;
	cudaMalloc((void**)&cArr, sizeof(int)*(threads_size + 1));
	cudaMemcpy(cArr, arr, sizeof(int)*(threads_size + 1), cudaMemcpyHostToDevice);
	free(arr);

	// save pinch
	this->pinch = screen->pixels->pitch;

	// save host pixels
	this->h_pixels = screen->pixels->pixels;

	//alloc device pixels
	int size = height * pinch;
	cudaMalloc((void**)&this->d_pixels, size);
	cudaMemcpy(this->d_pixels, this->h_pixels, size, cudaMemcpyHostToDevice);
}

gl::RenderOnGPU::~RenderOnGPU()
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
	cudaFree(d_pixels);
}

//==============================================================================================================
struct Shader : public gl::IShader {
	mat<2, 3, float> *varying_uv;  // triangle uv coordinates, written by the vertex shader, read by the fragment shader
	mat<4, 3, float> *varying_tri; // triangle coordinates (clip coordinates), written by VS, read by FS
	mat<3, 3, float> *varying_nrm; // normal per vertex to be interpolated by FS
	mat<3, 3, float> *ndc_tri;     // triangle in normalized device coordinates

	gl::ModelBuffer *model;
	Matrix Projection;
	Matrix ModelView;
	Vec3f light_dir;

	__device__ Shader(gl::ModelBuffer *mb, Matrix &Projection, Matrix &ModelView, Vec3f &light_dir)
	{
		this->model = mb;
		this->Projection = Projection;
		this->ModelView = ModelView;
		this->light_dir = light_dir;

		//cudaMalloc((void**)&varying_uv, sizeof(mat<2, 3, float>));
		//cudaMalloc((void**)&varying_tri, sizeof(mat<4, 3, float>));
		//cudaMalloc((void**)&varying_nrm, sizeof(mat<3, 3, float>));
		//cudaMalloc((void**)&ndc_tri, sizeof(mat<3, 3, float>));
		mat<2, 3, float> varying_uv;  // triangle uv coordinates, written by the vertex shader, read by the fragment shader
		mat<4, 3, float> varying_tri; // triangle coordinates (clip coordinates), written by VS, read by FS
		mat<3, 3, float> varying_nrm; // normal per vertex to be interpolated by FS
		mat<3, 3, float> ndc_tri;     // triangle in normalized device coordinates
		this->varying_uv = &varying_uv;
		this->varying_tri = &varying_tri;
		this->varying_nrm = &varying_nrm;
		this->ndc_tri = &ndc_tri;
	}

	__device__ virtual Vec4f vertex(int iface, int nthvert) {
		Vec2f uvRes = model->uv(iface, nthvert);
		varying_uv->set_col(nthvert, &uvRes);
		vec<3, float> proj1Res = proj<3>(((Projection*ModelView).invert_transpose()*embed<4>(model->normal(iface, nthvert), 0.f)));
		varying_nrm->set_col(nthvert, &proj1Res);
		Vec4f gl_Vertex = Projection * ModelView*embed<4>(model->vert(iface, nthvert));
		varying_tri->set_col(nthvert, &gl_Vertex);
		vec<3, float> proj2Res = proj<3>(gl_Vertex / gl_Vertex[3]);
		ndc_tri->set_col(nthvert, &proj2Res);
		return gl_Vertex;
	}

	__device__ virtual bool fragment(Vec3f bar, gl::Color::Device &color) {
		Vec3f bn = (*varying_nrm*bar).normalize();
		Vec2f uv = *varying_uv * bar;

		mat<3, 3, float> A;
		A[0] = ndc_tri->col(1) - ndc_tri->col(0);
		A[1] = ndc_tri->col(2) - ndc_tri->col(0);
		A[2] = bn;

		mat<3, 3, float> AI = A.invert();

		Vec3f i = AI * Vec3f((*varying_uv)[0][1] - (*varying_uv)[0][0], (*varying_uv)[0][2] - (*varying_uv)[0][0], 0);
		Vec3f j = AI * Vec3f((*varying_uv)[1][1] - (*varying_uv)[1][0], (*varying_uv)[1][2] - (*varying_uv)[1][0], 0);

		mat<3, 3, float> B;
		B.set_col(0, &i.normalize());
		B.set_col(1, &j.normalize());
		B.set_col(2, &bn);

		Vec3f n = (B*model->normal(uv)).normalize();

		float diff = fmaxf(0.f, n*light_dir);
		color = model->diffuse(uv)*diff;

		return false;
	}
};
//==============================================================================================================

__device__ void part(void* pixels, int pinch, int width, int height, gl::ModelBuffer *mb,
	int first, int last, float *zbuffer, float ra, float command, gl::RenderMode mode)
{
	Vec3f light_dir(1, 1, 1);
	Vec3f       eye(command, ra, 1);
	Vec3f    center(0, 0, 0);
	Vec3f        up(0, 1, 0);

	Matrix ModelView;
	Matrix Viewport;
	Matrix Projection;

	gl::camera::lookat(ModelView, eye, center, up);
	gl::camera::viewport(Viewport, width / 8, height / 8, width * 3 / 4, height * 3 / 4);
	gl::camera::projection(Projection, -1.f / (eye - center).norm());
	light_dir = proj<3>((Projection*ModelView*embed<4>(light_dir, 0.f))).normalize();

	if (mode == gl::RenderMode::Shaders)
	{
		//ModelBuffer *mb, Matrix &Projection, Matrix &ModelView, Vec3f &light_dir
		//triangle_s(mat<4, 3, float> &clipc, IShader &shader, void* pixels, int pinch, float *zbuffer, Matrix &Viewport)
		Shader shader(mb, Projection, ModelView, light_dir);
		for (int i = first; i < last; i++) {
			for (int j = 0; j < 3; j++) {
				shader.vertex(i, j);
			}
			////	//triangle(shader.varying_tri, shader, frame, zbuffer);
			gl::draw::triangle_s(shader.varying_tri, &shader, pixels, pinch, zbuffer, Viewport, ra);
		}
	}
	else if (mode == gl::RenderMode::Filled)
	{
		for (int i = first; i < last; i++) {
			Vec2i screen_coords[3];
			for (int j = 0; j < 3; j++) {
				Vec3f v = mb->vert(mb->face(i, j));
				Matrix result = Viewport * Projection*ModelView*Matrix(v);
				screen_coords[j] = Vec2f(result[0][0] / result[3][0], result[1][0] / result[3][0]);
			}
			gl::draw::triangle(screen_coords[0], screen_coords[1], screen_coords[2], pixels, pinch, &gl::Color::Device(255, 0, 0));
		}
	}
	else if (mode == gl::RenderMode::Wire)
	{
		for (int i = first; i < last; i++) {
			Vec3i screen_coords[3];
			for (int j = 0; j < 3; j++) {
				Vec3f v = mb->vert(mb->face(i, j));
				Matrix result = Viewport*Projection*ModelView*Matrix(v);
				screen_coords[j] = Vec3f(result[0][0]/result[3][0], result[1][0]/result[3][0], result[2][0]/result[3][0]);
			}
			gl::draw::line(screen_coords[0].x, screen_coords[0].y, screen_coords[1].x, screen_coords[1].y, pixels, pinch, &gl::Color::Device(255, 0, 0));
			gl::draw::line(screen_coords[1].x, screen_coords[1].y, screen_coords[2].x, screen_coords[2].y, pixels, pinch, &gl::Color::Device(0, 255, 0));
			gl::draw::line(screen_coords[2].x, screen_coords[2].y, screen_coords[0].x, screen_coords[0].y, pixels, pinch, &gl::Color::Device(0, 0, 255));
		}
	}
	else if (mode == gl::RenderMode::ShadersWithWire)
	{
		//ModelBuffer *mb, Matrix &Projection, Matrix &ModelView, Vec3f &light_dir
		//triangle_s(mat<4, 3, float> &clipc, IShader &shader, void* pixels, int pinch, float *zbuffer, Matrix &Viewport)
		Shader shader(mb, Projection, ModelView, light_dir);
		for (int i = first; i < last; i++) {
			Vec3i screen_coords[3];
			for (int j = 0; j < 3; j++) {
				shader.vertex(i, j);
				Vec3f v = mb->vert(mb->face(i, j));
				Matrix result = Viewport * Projection*ModelView*Matrix(v);
				screen_coords[j] = Vec3f(result[0][0] / result[3][0], result[1][0] / result[3][0], result[2][0] / result[3][0]);
			}
			////	//triangle(shader.varying_tri, shader, frame, zbuffer);
			gl::draw::triangle_s(shader.varying_tri, &shader, pixels, pinch, zbuffer, Viewport, ra);

			gl::draw::line(screen_coords[0].x, screen_coords[0].y, screen_coords[1].x, screen_coords[1].y, pixels, pinch, &gl::Color::Device(255, 0, 0));
			gl::draw::line(screen_coords[1].x, screen_coords[1].y, screen_coords[2].x, screen_coords[2].y, pixels, pinch, &gl::Color::Device(0, 255, 0));
			gl::draw::line(screen_coords[2].x, screen_coords[2].y, screen_coords[0].x, screen_coords[0].y, pixels, pinch, &gl::Color::Device(0, 0, 255));
		}
	}
}

int* gl::splitByThreads(int model, int parts)
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

__global__ void SplitByMPs(void* pixels, int pinch, int width, int height, gl::ModelBuffer *mb, int threads_size,
	int *arr, float *zbuffer, float ra, float command, gl::RenderMode mode)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//printf("size=%d\n", threads_size);
	
	if (idx < threads_size + 1)
	{
		//debugPrint(arr, threads_size + 1);
		//printf("idx=%d\n", idx);
		part(pixels, pinch, width, height, mb, arr[idx], arr[idx + 1], zbuffer, ra, command, mode);
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

void gl::RenderOnGPU::refresh(float direction, float command, RenderMode mode)
{
	clock_t begin = clock();

	cudaMemcpy(this->d_pixels, this->h_pixels, height * pinch, cudaMemcpyHostToDevice);
	cudaMemcpy(zBufferGPU, zbuffer, width*height * sizeof(float), cudaMemcpyHostToDevice);

	SplitByMPs <<<128, 128 >>> (this->d_pixels, pinch, width, height, model, threads_size, cArr, zBufferGPU, direction, command, mode);

	//printf("model=%d, threads_size=%d\n",m->nfaces(), threads_size);
	//RenderSadersMode <<<128, 128 >>> (this->d_pixels, pinch, width, height, model, threads_size, cArr, zBufferGPU, direction, command);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cudaMemcpy(this->h_pixels, this->d_pixels, height * pinch, cudaMemcpyDeviceToHost);

	clock_t end = clock();
	this->renderFrameTime = float(end - begin) / CLOCKS_PER_SEC;
	printf("time: %lf\n", this->renderFrameTime);
}

float& gl::RenderOnGPU::GetRenderFrameTime()
{
	return this->renderFrameTime;
}