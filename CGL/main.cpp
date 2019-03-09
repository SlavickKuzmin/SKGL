#include <algorithm>
#include "model.h"
#include <SDL.h>
#undef main

#include "Helpers.cuh"

#include "Screen.h"

const int width = 800;
const int height = 800;
void runRender();
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
	//for(int i = 0; i < height; i++)
	//		screen->setPixel(i, height-i, 0xFFFF0000);

	Model *model = new Model("E:\\Diplom\\SDL\\CGL\\obj\\diablo3_pose\\diablo3_pose.obj");

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
		//runKernel(screen->pixels->pixels, screen->pixels->pitch, screen->width, screen->height);
		drawModel(screen->pixels->pixels, screen->pixels->pitch, screen->width, screen->height, model);
		screen->setScreen(window);
	}


	delete model;
	delete screen;
	SDL_DestroyWindow(window);
	SDL_Quit();
}