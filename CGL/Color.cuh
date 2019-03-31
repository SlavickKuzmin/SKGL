#pragma once
#include "SDL.h"
#include "cuda_runtime_api.h"

#define ColorByte unsigned char

struct Color
{
	ColorByte red;
	ColorByte green;
	ColorByte blue;
	ColorByte alpha;
};

__device__ Uint32 packRGBAToUint32(ColorByte r, ColorByte g, ColorByte b, ColorByte a);
__device__ Uint32 packColorToUint32(Color *color);
__device__ Color unpackUint32ToColor(Uint32 color);