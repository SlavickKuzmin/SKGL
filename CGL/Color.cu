#include "Color.cuh"

__device__ Uint32 packRGBAToUint32(ColorByte r, ColorByte g, ColorByte b, ColorByte a)
{
	return (a << 24) | (b << 16) | (g << 8) | r;
}

__device__ Uint32 packColorToUint32(Color *color)
{
	return ((*color).alpha << 24) | ((*color).blue << 16) | ((*color).green << 8) | (*color).red;
}

__device__ Color unpackUint32ToColor(Uint32 col)
{
	Color color;
	color.red = (*((Uint32*)col)) & 0xFF;
	color.green = (*((Uint32*)col) >> 8) & 0xFF;
	color.blue = (*((Uint32*)col) >> 16) & 0xFF;
	color.alpha = (*((Uint32*)col) >> 24) & 0xFF;
	return color;
}