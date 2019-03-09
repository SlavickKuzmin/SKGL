#pragma once

#include "SDL.h"

class Screen
{
public:
	Screen(int width, int height);
	~Screen();
	void setPixel(int x, int y, Uint32 color);
	void setScreen(SDL_Window *window);
	SDL_Surface *pixels;
	int width;
	int height;
};