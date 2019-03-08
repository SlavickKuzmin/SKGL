#pragma once
#include "SDL.h"

#define ColorByte unsigned char

struct Color
{
	Color(ColorByte r, ColorByte g, ColorByte b, ColorByte a);
	Color() = default;
	ColorByte red;
	ColorByte green;
	ColorByte blue;
	ColorByte alpha;
	Uint32 getColor();
};