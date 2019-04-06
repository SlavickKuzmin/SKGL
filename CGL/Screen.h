#ifndef _SCREEN_H_
#define _SCREEN_H_



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

#endif // !_SCREEN_H_