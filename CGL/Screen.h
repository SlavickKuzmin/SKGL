#pragma once

#include "SDL.h"
#include "Color.h"

class Screen
{
public:
	Screen(int width, int height);
	~Screen();
	void setPixel(int x, int y, Color& color);
	Color getPixel(int x, int y);
	void setScreen(SDL_Window *window);
private:
	SDL_Surface *pixels;
	int width;
	int height;
};