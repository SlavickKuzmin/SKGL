#include "refreshOnCPU.h"


RenderOnCpu::RenderOnCpu(Model *model, int width, int height, SDL_Renderer *renderer)
{
	this->width = width;
	this->height = height;
	
	// make model
	this->model = model;

	this->renderer = renderer;

	// init z-buffer
	this->zbuffer = new float[width*height];
	for (int i = width * height; i--; this->zbuffer[i] = -std::numeric_limits<float>::max());

	this->light_dir = Vec3f(1, 1, 1);
	this->eye = Vec3f(1, 1, 3);
	this->center = Vec3f(0, 0, 0);
	this->up = Vec3f(0, 5, 100);

	// init lookat, viewport, proj matrix and light dir
	// TODO: move in refresh method
	lookat(eye, center, up);
	viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
	projection(-1.f / (eye - center).norm());
	light_dir = proj<3>((Projection*ModelView*embed<4>(light_dir, 0.f))).normalize();
}
RenderOnCpu::~RenderOnCpu()
{
	delete[] this->zbuffer;
}

void RenderOnCpu::setShader(Shader *shader)
{
	this->shader = shader;
}

Model* RenderOnCpu::getModel()
{
	return this->model;
}

Vec3f* RenderOnCpu::getLight_dir()
{
	return &(this->light_dir);
}

void RenderOnCpu::refresh()
{
	printf("start-");
	startTime = clock();

	for (int i = 0; i < model->nfaces(); i++) {
		for (int j = 0; j < 3; j++) {
			this->shader->vertex(i, j);
		}
		triangle(this->shader->varying_tri, *shader, this->renderer, zbuffer);
	}

	endTime = clock();
	elapsedSecs = double(endTime - startTime) / CLOCKS_PER_SEC;
	printf("end: %lf\n", elapsedSecs);
}