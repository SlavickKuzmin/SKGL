#pragma once

#include <string>
#include <stdio.h>
#include "model.h"
#include "geometry.h"
#include "our_gl.h"
#include <vector>
#include <limits>
#include <algorithm>
#include "Shader.h"

#include <SDL.h>
#undef main

#include <ctime> // for time measure

class RenderOnCpu
{
public:
	RenderOnCpu(Model *model, int width, int height, SDL_Renderer *renderer);
	~RenderOnCpu();
	void setShader(Shader *shader);
	void refresh();
	Model* getModel();
	Vec3f* getLight_dir();
private:
	Shader *shader;
	float *zbuffer;
	SDL_Renderer *renderer;
	Model *model;
	clock_t startTime;
	clock_t endTime;
	double elapsedSecs;
	int width;
	int height;
	Vec3f light_dir;
	Vec3f       eye;
	Vec3f    center;
	Vec3f        up;
};