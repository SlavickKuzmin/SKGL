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
	gl::Screen *screen = new gl::Screen(width, height);
	//for(int i = 0; i < height; i++)
	//		screen->setPixel(i, height-i, 0xFFFF0000);

	Model *model = new Model("E:\\Diplom\\SDL\\CGL\\obj\\diablo3_pose\\diablo3_pose.obj");
	TGAImage *background = new TGAImage();
	background->read_tga_file("E:\\Diplom\\sky.tga");
	//Model *model = new Model("E:\\Diplom\\SDL\\CGL\\obj\\african_head\\african_head.obj");
	//Model *model = new Model("E:\\3d\\stature\\statue.obj");
	gl::RenderOnGPU *render = new gl::RenderOnGPU(model, screen);

	bool quit = false;

	//Event handler
	SDL_Event e;

	float command = 1;
	float direction = 1;

	char title[5];
	gl::RenderMode mode = gl::RenderMode::Shaders;
	bool UseBackground = false;
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
					direction += 0.0374532925f;
					break;

				case SDLK_DOWN:
					printf("down=%f\n", direction);
					direction += -0.0374532925f;
					break;

				case SDLK_LEFT:
					printf("left=%f\n", command);
					command += 0.0374532925f;
					break;

				case SDLK_RIGHT:
					printf("left=%f\n", command);
					command += -0.0374532925f;
					break;
				case SDLK_1:
					mode = gl::RenderMode::Shaders;
					break;
				case SDLK_2:
					mode = gl::RenderMode::Filled;
					break;
				case SDLK_3:
					mode = gl::RenderMode::Wire;
					break;
				case SDLK_4:
					mode = gl::RenderMode::ShadersWithWire;
					break;
				case SDLK_q:
					UseBackground = !UseBackground;
					break;
				default:
					break;
				}
			}
		}
		
		screen->ClearScreen(gl::Color::Host(0,0,0));
		//SetPixel(screen, 100, 100, Color::Host(255, 0, 0));
		//SetLine(screen, 120, 120, 200, 200, Color::Host(0,255,0));
		//SetTriangle(screen, 10, 10, 30, 30, 30, 10, Color::Host(0,0,255));
		//SetRectangle(screen, 10, 300, 10, 400, 100, 400, 100, 300, Color::Host(255, 0,0));
		//SetCircle(screen, 60, 60, 20, Color::Host(255, 0, 255));
		//int coords[12] = {0,20, 500, 200, 10, 20, 125, 600, 200, 300, 2, 8};
		//SetPolygon(screen, coords, 12, Color::Host(255, 255, 0));
		if (UseBackground)
		{
			gl::draw::SetImage(screen, background);
		}
		render->refresh(direction, command, mode);

		// FPS count
		int fps = 1.0f / render->GetRenderFrameTime();
		sprintf(title, "FPS: %d", fps);
		SDL_SetWindowTitle(window, title);

		screen->setScreen(window);
	}

	delete screen;
	delete model;
	delete background;

	delete render;
	SDL_DestroyWindow(window);
	SDL_Quit();
}