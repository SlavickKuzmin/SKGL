#include "model.h"
#include "SDL.h"
#include "Shader.cuh"
#include "cuda_geometry.cuh"
#include "cuda_gl.cuh"
#include "ModelBuffer.cuh"

class RenderOnGPU
{
public:
	RenderOnGPU(Model *model, int width, int height);
	~RenderOnGPU();
	void setShader(Shader *shader);
	void refresh(void* pixels, int pinch, int width, int height);
	__device__ void devInit();
	Model* getModel();
	Vec3f* getLight_dir();
private:
	Shader *shader;
	float *zbuffer;
	SDL_Renderer *renderer;
	Model *model;
	int width;
	int height;
	Vec3f light_dir;
	Vec3f       eye;
	Vec3f    center;
	Vec3f        up;
	Matrix Viewport;
	Matrix Projection;
	Matrix ModelView;
};