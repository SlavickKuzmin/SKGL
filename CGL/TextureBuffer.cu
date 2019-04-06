#include "TextureBuffer.cuh"

TextureBuffer::TextureBuffer(TGAImage &image)
{
	cudaMalloc((void**)&d_pWidth, sizeof(int));
	cudaMalloc((void**)&d_pHeight, sizeof(int));
	cudaMalloc((void**)&d_pBytesApp, sizeof(int));

	cudaMemcpy(d_pWidth, &(image.width), sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pHeight, &(image.height), sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pBytesApp, &(image.bytespp), sizeof(int), cudaMemcpyHostToDevice);

	unsigned long nbytes = image.width * image.height*image.bytespp;
	cudaMalloc((void**)&texture_binary_data, nbytes);

	cudaMemcpy(texture_binary_data, image.data, nbytes, cudaMemcpyHostToDevice);
	printf("Texture constructor called\n");
}

TextureBuffer::~TextureBuffer()
{
	cudaFree(texture_binary_data);
	cudaFree(d_pWidth);
	cudaFree(d_pHeight);
	cudaFree(d_pBytesApp);
	printf("Texture destructor called\n");
}

__device__ int TextureBuffer::getWidth()
{
	return *(this->d_pWidth);
}
__device__ int TextureBuffer::getHeight()
{
	return *(this->d_pHeight);
}
__device__ int TextureBuffer::getBytesApp()
{
	return *(this->d_pBytesApp);
}