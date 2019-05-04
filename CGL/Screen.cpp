#include "Screen.h"
#include <stdio.h>

// Constructor: create a surface with given width and heigth.
gl::Screen::Screen(int width, int height)
{
	// Sets a mask in depends of PC bit ordering.
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
	// Allocate a SDL surface.
	pixels = SDL_CreateRGBSurface(SDL_SWSURFACE, width, height, 32,
		rmask, gmask, bmask, amask);

	if (pixels == NULL) {
		fprintf(stderr, "CreateRGBSurface failed: %s\n", SDL_GetError());
		exit(1);
	}

	// Set current width and height.
	this->width = width;
	this->height = height;
}

// Default destructor.
gl::Screen::~Screen()
{
	// Free a allocated SDL surface.
	SDL_FreeSurface(pixels);
}

// Set a pixel by {x,y} position in given color.
void gl::Screen::setPixel(int x, int y, Uint32 color)
{
	Uint8 *pixel = (Uint8*)pixels->pixels;
	pixel += (y * pixels->pitch) + (x * sizeof(Uint32));
	*((Uint32*)pixel) = color;//abgr
}

// Set screen to given window.
void gl::Screen::setScreen(SDL_Window *window)
{
	SDL_BlitSurface(pixels, 0, SDL_GetWindowSurface(window), 0);
	SDL_UpdateWindowSurface(window);
}

// Clear screen with given color.
void gl::Screen::ClearScreen(gl::Color::Host &ClearColor)
{
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
			this->setPixel(i, j, gl::Color::getUint32FromhColor(&ClearColor));
}

// Gets a current screen width.
int gl::Screen::GetWidth() const
{
	return this->width;
}

// Gets a current screen height.
int gl::Screen::GetHeight() const
{
	return this->height;
}