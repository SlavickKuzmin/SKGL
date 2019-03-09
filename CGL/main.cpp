#include <algorithm>
#include "refreshOnCPU.h"
#include "model.h"
#include <SDL.h>
#undef main

#include "refreshOnGPU.cuh"

#include "Screen.h"

const int width = 800;
const int height = 800;

void runGPURender();

int main() {
	//runRender();
	runGPURender();
	getchar();
	return EXIT_SUCCESS;
}

void runGPURender()
{
	// SDL main loop
	SDL_Event event;
	SDL_Window *window;

	SDL_Init(SDL_INIT_VIDEO);
	window = SDL_CreateWindow("GPU render", 50, 50, width, height,
		SDL_WINDOW_SHOWN);
	
	// my inits
	Screen *screen = new Screen(width, height);
	for(int i = 0; i < height; i++)
			screen->setPixel(i, height-i, 0xFFFF0000);

	bool quit = false;

	//Event handler
	SDL_Event e;

	//While application is running
	while (!quit)
	{
		//Handle events on queue
		while (SDL_PollEvent(&e) != 0)
		{
			//User requests quit
			if (e.type == SDL_QUIT)
			{
				quit = true;
			}
		}
		runKernel(screen->pixels->pixels, screen->pixels->pitch, screen->width, screen->height);
		screen->setScreen(window);
	}

	SDL_DestroyWindow(window);
	SDL_Quit();
}

void runRender()
{
	// SDL main loop
	SDL_Event event;
	SDL_Renderer *renderer;
	SDL_Window *window;

	SDL_Init(SDL_INIT_VIDEO);
	SDL_CreateWindowAndRenderer(width, height, 0, &window, &renderer);
	SDL_RenderClear(renderer);

	Model *model = new Model("E:\\Diplom\\SDL\\CGL\\obj\\diablo3_pose\\diablo3_pose.obj");
	RenderOnCpu *cpuRenderer = new RenderOnCpu(model, width, height, renderer);

	Shader *shader = new Shader(cpuRenderer->getModel(), cpuRenderer->getLight_dir());
	cpuRenderer->setShader(shader);

	bool quit = false;

	//Event handler
	SDL_Event e;

	//While application is running
	while (!quit)
	{
		//Handle events on queue
		while (SDL_PollEvent(&e) != 0)
		{
			//User requests quit
			if (e.type == SDL_QUIT)
			{
				quit = true;
			}
		}

		cpuRenderer->refresh();
		SDL_RenderPresent(renderer);
	}

	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();

	delete cpuRenderer;
	delete model;
	delete shader;
}