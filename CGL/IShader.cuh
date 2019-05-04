#ifndef _ISHADER_CUH_
#define _ISHADER_CUH_

#include "cuda_geometry.cuh"
#include "Color.cuh"

namespace gl
{
	using namespace gl::computing;

	// Represents a shaders mechanism.
	struct IShader {
		virtual __device__ Vec4f vertex(int iface, int nthvert) = 0;
		virtual __device__ bool fragment(Vec3f bar, gl::Color::Device &color) = 0;
	};
}

#endif //_ISHADER_CUH_