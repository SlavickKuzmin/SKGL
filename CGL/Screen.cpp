#include "Screen.h"
#include <stdio.h>

Screen::Screen(int width, int height)
{
	Uint32 rmask, gmask, bmask, amask;
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
	rmask = 0xff000000;
	gmask = 0x00ff0000;
	bmask = 0x0000ff00;
	amask = 0x000000ff;
#else
	rmask = 0x000000ff;
	gmask = 0x0000ff00;
	bmask = 0x00ff0000;
	amask = 0xff000000;
#endif
	pixels = SDL_CreateRGBSurface(SDL_SWSURFACE, width, height, 32,
		rmask, gmask, bmask, amask);

	if (pixels == NULL) {
		fprintf(stderr, "CreateRGBSurface failed: %s\n", SDL_GetError());
		exit(1);
	}

	this->width = width;
	this->height = height;
}

Screen::~Screen()
{
	SDL_FreeSurface(pixels);
}

void Screen::setPixel(int x, int y, Uint32 color)
{
	Uint8 *pixel = (Uint8*)pixels->pixels;
	pixel += (y * pixels->pitch) + (x * sizeof(Uint32));
	*((Uint32*)pixel) = color;//abgr
}

void Screen::setScreen(SDL_Window *window)
{
	SDL_BlitSurface(pixels, 0, SDL_GetWindowSurface(window), 0);
	SDL_UpdateWindowSurface(window);
}