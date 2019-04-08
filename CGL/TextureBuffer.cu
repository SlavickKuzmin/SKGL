/*
 * Description: Contains a API implementation for loading texture image
 * in GPU memory and manupulation with it.
 * Created by Viacheslav Kuzmin 2019
 */

#include "TextureBuffer.cuh"

// Construct a class from image stored on CPU memory, 
// allocate GPU memory.
TextureBuffer::TextureBuffer(TGAImage &image)
{
	// Allocate device memory for data storage.
	cudaMalloc((void**)&d_pWidth, sizeof(int));
	cudaMalloc((void**)&d_pHeight, sizeof(int));
	cudaMalloc((void**)&d_pBytesApp, sizeof(int));

	// Copy memory from host to device.
	cudaMemcpy(d_pWidth, &(image.width), sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pHeight, &(image.height), sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pBytesApp, &(image.bytespp), sizeof(int), cudaMemcpyHostToDevice);

	unsigned long nbytes = image.width * image.height*image.bytespp;
	cudaMalloc((void**)&texture_binary_data, nbytes);

	cudaMemcpy(texture_binary_data, image.data, nbytes, cudaMemcpyHostToDevice);
}

// Free all used GPU memory.
TextureBuffer::~TextureBuffer()
{
	// Free all alocated device memory.
	cudaFree(texture_binary_data);
	cudaFree(d_pWidth);
	cudaFree(d_pHeight);
	cudaFree(d_pBytesApp);
}

// Gets a texture width.
__device__ int TextureBuffer::getWidth()
{
	return *(this->d_pWidth);
}

// Gets a texture height.
__device__ int TextureBuffer::getHeight()
{
	return *(this->d_pHeight);
}

// Gets a texture bytes app.
__device__ int TextureBuffer::getBytesApp()
{
	return *(this->d_pBytesApp);
}

// Gets a texture pixel color from specific position (x and y coords).
__device__ Color TextureBuffer::get(int x, int y)
{
	// Validate input parameters.
	if (!texture_binary_data || x < 0 || y < 0 || x >= getWidth() || y >= getHeight()) {
		return Color();
	}

	// return a requested color.
	return Color(texture_binary_data + (x + y * getWidth())*getBytesApp(), getBytesApp());
}