#ifndef _RASTERISER_CUH_
#define _RASTERISER_CUH_

#include "cuda_geometry.cuh"
#include <float.h> // for max min values
#include "Color.cuh"

namespace gl
{
	namespace camera
	{
		using namespace gl::computing;

		// Set to given viewport to given parameters.
		__device__ void viewport(Matrix &Viewport, int x, int y, int w, int h);

		// Set a coeficient to given projection.
		__device__ void projection(Matrix &Projection, float coeff); // coeff = -1/c

		// Set current camera position to model view.
		__device__ void lookat(Matrix &ModelView, Vec3f eye, Vec3f center, Vec3f up);

		// Barycentric coordinates transform.
		__device__ Vec3f barycentric(Vec2f A, Vec2f B, Vec2f C, Vec2f P);
	}
}

#endif // !_RASTERISER_CUH_
