#ifndef __IMAGE_BUFFER_CUH__
#define __IMAGE_BUFFER_CUH__

#include <fstream>
#include "cuda_runtime_api.h"
#include "tgaimage.h"

struct TGAColorBuf {
	unsigned char *bgra[4];
	unsigned char bytespp;

	__device__ TGAColorBuf() : bgra(), bytespp(1) {
		for (int i = 0; i < 4; i++) bgra[i] = 0;
	}

	__device__ TGAColorBuf(unsigned char R, unsigned char G, unsigned char B, unsigned char A = 255) : bgra(), bytespp(4) {
		*bgra[0] = B;
		*bgra[1] = G;
		*bgra[2] = R;
		*bgra[3] = A;
	}

	__device__ TGAColorBuf(unsigned char v) : bgra(), bytespp(1) {
		for (int i = 0; i < 4; i++) *bgra[i] = 0;
		*bgra[0] = v;
	}

	__device__ TGAColorBuf(const unsigned char *p, unsigned char bpp) : bgra(), bytespp(bpp) {
		for (int i = 0; i < (int)bpp; i++) {
			*bgra[i] = p[i];
		}
		for (int i = bpp; i < 4; i++) {
			*bgra[i] = 0;
		}
	}

	__device__ unsigned char& operator[](const int i) { return *bgra[i]; }

	__device__ TGAColorBuf operator *(float intensity) const {
		TGAColorBuf res = *this;
		intensity = (intensity > 1.f ? 1.f : (intensity < 0.f ? 0.f : intensity));
		for (int i = 0; i < 4; i++) *(res.bgra[i]) = *bgra[i] * intensity;
		return res;
	}
};

class TGAImageBuf {
protected:
	 unsigned char* data;
	 int width;
	 int height;
	 int bytespp;
public:
	enum Format {
		GRAYSCALE = 1, RGB = 3, RGBA = 4
	};

	 TGAImageBuf(TGAImage *image);
	__device__ bool flip_horizontally();
	__device__ TGAColorBuf get(int x, int y);
	__device__ bool set(int x, int y, TGAColorBuf &c);
	__device__ bool set(int x, int y, const TGAColorBuf &c);
	 ~TGAImageBuf();
	__device__ int get_width();
	__device__ int get_height();
	__device__ int get_bytespp();
	__device__ unsigned char *buffer();
	__device__ void clear();
};

#endif //__IMAGE_BUFFER_CUH__
