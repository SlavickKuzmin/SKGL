#ifndef _COLOR_CUH_
#define _COLOR_CUH_

#include "SDL.h"
#include "cuda_runtime_api.h"

namespace gl
{
	/*
	 Namespase that contains a color classes definitions
	 for host and device code.
	 Classes implemets a basic constructors and pack\unpack
	 operation for SDL library.
	*/
	namespace Color
	{

		// Define a user-friendly color type.
		typedef unsigned char Channel;

		// Represents a device executable color object.
		struct Device
		{
			// RGBA channels definitions.
			Channel red;
			Channel green;
			Channel blue;
			Channel alpha;

			// Default constructor, by default color is black.
			__device__ Device()
			{
				this->blue = 0; this->green = 0; this->red = 0;
				this->alpha = 255;
			}

			// Constructor from RGB channels.
			// Notice: aplha chnnel is always 255.
			__device__ Device(Channel r, Channel g, Channel b)
			{
				this->blue = b; this->green = g; this->red = r;
				this->alpha = 255;
			}

			// Contructor for pointer.
			__device__ Device(const unsigned char *p, unsigned char bpp)
			{
				this->blue = p[0]; this->green = p[1]; this->red = p[2];
				if (bpp > 4)
				{
					this->alpha = p[3];
				}
			}

			// Index operation for color. 
			//NOTICE: color stored in BGRA format!
			__device__ unsigned char& operator[](const int i)
			{
				switch (i)
				{
				case 0:
					return this->blue;
				case 1:
					return this->green;
				case 2:
					return this->red;
				case 3:
					return this->alpha;
				default:
					return this->alpha;
				}
			}

			// Multiplication operator overload (for change color intensivity).
			__device__ Device operator *(float intensity) const;
		};

		// Pack RGBA color passed by channels to uint32 bit format.
		__device__ Uint32 packRGBAToUint32(Channel r, Channel g, Channel b, Channel a);

		// Pack from device RGBA color class to uint32 bit format.
		__device__ Uint32 packColorToUint32(Device *color);

		// UnPack a unit32 bit format to device RGBA color class.
		__device__ Device unpackUint32ToColor(Uint32 color);

		// Represents a host executable color object.
		struct Host
		{
			// RGBA channels definitions.
			Channel red;
			Channel green;
			Channel blue;
			Channel alpha;

			// Default constructor, by default color is black.
			__host__ Host()
			{
				this->blue = 0; this->green = 0; this->red = 0;
				this->alpha = 255;
			}

			// Constructor from RGBA channels.
			__host__ Host(Channel r, Channel g, Channel b, Channel a)
			{
				this->blue = b; this->green = g; this->red = r;
				this->alpha = a;
			}

			// Constructor from RGB channels.
			// Notice: aplha chnnel is always 255.
			__host__ Host(Channel r, Channel g, Channel b)
			{
				this->blue = b; this->green = g; this->red = r;
				this->alpha = 255;
			}
		};

		// Pack from RGBA channels to uint32 bit format.
		__host__ Uint32 getUint32FromRGBA(Channel r, Channel g, Channel b, Channel a);

		// Pack from host RGBA color class to uint32 bit format.
		__host__ Uint32 getUint32FromhColor(Host *color);

		// UnPack a unit32 bit format to host RGBA color.
		__host__ Host gethColorFromUint32(Uint32 color);

	}
}
#endif // !_COLOR_CUH_