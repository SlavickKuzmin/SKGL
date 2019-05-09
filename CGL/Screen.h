#ifndef _SCREEN_H_
#define _SCREEN_H_

#include "SDL.h"
#include "Color.cuh"

namespace gl
{

	/*
	 A screen helper class, represents a window with given Width and Height,
	 allow set and get pixel on screen in given area, clear screen by given
	 color.
	*/
	class Screen
	{
	public:
		// Constructor: create a surface with given width and heigth.
		Screen(int width, int height);

		// Default destructor.
		~Screen();

		// Set a pixel by {x,y} position in given color.
		void setPixel(int x, int y, Uint32 color);

		// Set screen to given window.
		void setScreen(SDL_Window *window);

		int  GetTextureScreen();

		// Clear screen with given color.
		void ClearScreen(gl::Color::Host &ClearColor);

		// Screen pixels storage structure.
		SDL_Surface *pixels;

		// Gets a current screen width.
		int GetWidth() const;

		// Gets a current screen height.
		int GetHeight() const;
	private:
		// Screen width.
		int width;

		// Screen height.
		int height;
	};
}
#endif // !_SCREEN_H_