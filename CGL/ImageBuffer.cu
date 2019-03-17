#include "ImageBuffer.cuh"
#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <math.h>
#include "tgaimage.h"

TGAImageBuf::TGAImageBuf(TGAImage *image) {
	this->width = image->width;
	this->height = image->height;
	this->bytespp = image->bytespp;
	//use cuda malloc
	cudaMalloc((void**)data, width*height*bytespp);
	//use cuda memcopy
	cudaMemcpy(data, image->data, width*height*bytespp, cudaMemcpyHostToDevice);
}

TGAImageBuf::~TGAImageBuf() {
	//use cudafree
	if (data) cudaFree(data);
}

__device__ TGAColorBuf TGAImageBuf::get(int x, int y) {
	if (!data || x < 0 || y < 0 || x >= width || y >= height) {
		return TGAColorBuf();
	}
	return TGAColorBuf(data + (x + y * width)*bytespp, bytespp);
}

__device__ bool TGAImageBuf::set(int x, int y, TGAColorBuf &c) {
	if (!data || x < 0 || y < 0 || x >= width || y >= height) {
		return false;
	}
	//add error checks
	memcpy(data + (x + y * width)*bytespp, c.bgra, bytespp);
	return true;
}

__device__ bool TGAImageBuf::set(int x, int y, const TGAColorBuf &c) {
	if (!data || x < 0 || y < 0 || x >= width || y >= height) {
		return false;
	}
	//add error check
	memcpy(data + (x + y * width)*bytespp, c.bgra, bytespp);
	return true;
}

__device__ int TGAImageBuf::get_bytespp() {
	return bytespp;
}

__device__ int TGAImageBuf::get_width() {
	return width;
}

__device__ int TGAImageBuf::get_height() {
	return height;
}

__device__ bool TGAImageBuf::flip_horizontally() {
	if (!data) return false;
	int half = width >> 1;
	for (int i = 0; i < half; i++) {
		for (int j = 0; j < height; j++) {
			TGAColorBuf c1 = get(i, j);
			TGAColorBuf c2 = get(width - 1 - i, j);
			set(i, j, c2);
			set(width - 1 - i, j, c1);
		}
	}
	return true;
}

__device__ unsigned char *TGAImageBuf::buffer() {
	return data;
}

__device__ void TGAImageBuf::clear() {
	//add error check
	memset((void *)data, 0, width*height*bytespp);
}