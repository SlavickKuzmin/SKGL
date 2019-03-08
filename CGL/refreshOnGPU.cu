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