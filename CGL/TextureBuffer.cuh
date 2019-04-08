/*
 * Description: Contains a API definition for loading texture image 
 * in GPU memory and manupulation with it.
 * Created by Viacheslav Kuzmin 2019
 */

#ifndef _TEXTURE_BUFFER_CUH_
#define _TEXTURE_BUFFER_CUH_

#include "tgaimage.h"
#include "Color.cuh"
#include "cuda_runtime_api.h"

/*
 * API definition for loading and work with textures
 * in GPU memory. 
*/
class TextureBuffer
{
public:
	// Construct a class from image stored on CPU memory, 
	// allocate GPU memory.
	TextureBuffer(TGAImage &image);

	// Free all used GPU memory.
	~TextureBuffer();

	// Binary data container, store all texture data (RGBA).
	unsigned char *texture_binary_data;

	// Device pointer to texture width.
	int *d_pWidth;

	// Device pointer to texture height.
	int *d_pHeight;

	// Device pointer to texture Bytes App.
	int *d_pBytesApp;

	// Gets a texture width.
	__device__ int getWidth();

	// Gets a texture height.
	__device__ int getHeight();

	// Gets a texture bytes app.
	__device__ int getBytesApp();

	// Gets a texture pixel color from specific position (x and y coords).
	__device__ Color get(int x, int y);
};

#endif // !_TEXTURE_BUFFER_CUH_
