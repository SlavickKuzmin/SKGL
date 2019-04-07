#include <algorithm>
#include "model.h"
#include <SDL.h>
#undef main

#include "RenderOnGPU.cuh"

#include "Screen.h"

#include <Windows.h>

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
	//Model *model = new Model("E:\\Diplom\\SDL\\CGL\\obj\\african_head\\african_head.obj");
	//Model *model = new Model("E:\\3d\\stature\\statue.obj");
	RenderOnGPU *render = new RenderOnGPU(model, width, height);

	bool quit = false;

	//Event handler
	SDL_Event e;

	float command = 1;
	float direction = 1;

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
			//User presses a key
			else if (e.type == SDL_KEYDOWN)
			{
				//Select surfaces based on key press
				switch (e.key.keysym.sym)
				{
				case SDLK_UP:
					printf("up=%f\n", direction);
					direction += 0.1f;
					break;

				case SDLK_DOWN:
					printf("down=%f\n", direction);
					direction += -0.1f;
					break;

				case SDLK_LEFT:
					printf("left=%f\n", command);
					command += 0.1f;
					break;

				case SDLK_RIGHT:
					printf("left=%f\n", command);
					command += -0.1f;
					break;

				default:
					break;
				}
			}
		}
		
		for (int i = 0; i < width; i++)
			for (int j = 0; j < height; j++)
				screen->setPixel(i, j, 0xFF000000);
		render->refresh(screen->pixels->pixels, screen->pixels->pitch, screen->width, screen->height, direction, command);
		//for (int i = 0; i < width; i++)
		//	for (int j = 0; j < height; j++)
		//		screen->setPixel(i, j, 0xFFFFFFFF);
		screen->setScreen(window);

		
	}

	delete screen;
	delete model;
	
	delete render;
	SDL_DestroyWindow(window);
	SDL_Quit();
}