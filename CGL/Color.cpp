#include "Color.h"

Color::Color(ColorByte r, ColorByte g, ColorByte b, ColorByte a)
{
	this->red = r;
	this->green = g;
	this->blue = b;
	this->alpha = a;
}

Uint32 Color::getColor()
{
	return (alpha << 24) | (blue << 16) | (green << 8) | red; // abgr
}