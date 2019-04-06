#ifndef _TEXTURE_BUFFER_CUH_
#define _TEXTURE_BUFFER_CUH_

#include "tgaimage.h"
#include "cuda_runtime_api.h"

class TextureBuffer
{
public:
	TextureBuffer(TGAImage &image);
	~TextureBuffer();
	unsigned char *texture_binary_data;
	int *d_pWidth;
	int *d_pHeight;
	int *d_pBytesApp;
	__device__ int getWidth();
	__device__ int getHeight();
	__device__ int getBytesApp();
};

#endif // !_TEXTURE_BUFFER_CUH_
