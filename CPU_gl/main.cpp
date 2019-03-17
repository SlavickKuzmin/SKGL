#include <algorithm>
#include "model.h"
#include <SDL.h>
#include "refreshOnCPU.h"
#include "Shader.h"
#undef main


const int width = 800;
const int height = 800;
void runRender();
void runGPURender();

int main() {
	runRender();
	return EXIT_SUCCESS;
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
	//Model *model = new Model("E:\\3d\\stature\\statue.obj");
	//Model *model = new Model("e:\\3d\\b2\\bird.obj");
	printf("ok\n");
	//getchar();
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