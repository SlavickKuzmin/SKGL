#ifndef _COLOR_CUH_
#define _COLOR_CUH_

#include "SDL.h"
#include "cuda_runtime_api.h"

#define ColorByte unsigned char

struct Color
{
	ColorByte red;
	ColorByte green;
	ColorByte blue;
	ColorByte alpha;
	__device__ Color()
	{
		this->blue = 0;
		this->green = 0;
		this->red = 0;
		this->alpha = 255;
	}
	__device__ Color(const unsigned char *p, unsigned char bpp)
	{
		this->blue = p[0];
		this->green = p[1];
		this->red = p[2];
		if (bpp < 4)
		{
			this->alpha = p[3];
		}		
		//for (int i = 0; i < (int)bpp; i++) {
		//	bgra[i] = p[i];
		//}
		//for (int i = bpp; i < 4; i++) {
		//	bgra[i] = 0;
		//}
	}
	__device__ unsigned char& operator[](const int i)
	{ 
		//return bgra[i];
		if (i == 0)
		{
			return this->blue;
		}
		else if (i == 1)
		{
			return this->green;
		}
		else if (i == 2)
		{
			return this->red;
		}
		else if (i == 3)
		{
			return this->alpha;
		}
		return this->alpha;
	}
	__device__ Color operator *(float intensity) const;
};

__device__ Uint32 packRGBAToUint32(ColorByte r, ColorByte g, ColorByte b, ColorByte a);
__device__ Uint32 packColorToUint32(Color *color);
__device__ Color unpackUint32ToColor(Uint32 color);

#endif // !_COLOR_CUH_