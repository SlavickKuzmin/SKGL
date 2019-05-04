/*
 * Description: Contains a API definition for loading .obj files from host structure
 * to GPU device memory, manipulating with structure and manage all fileds.
 * Created by Viacheslav Kuzmin 2019
 */

#ifndef _MODEL_BUFFER_CUH_
#define _MODEL_BUFFER_CUH_

#include "cuda_geometry.cuh"
#include "cuda_runtime_api.h"
#include "model.h"
#include "Color.cuh"
#include "TextureBuffer.cuh"

namespace gl
{

	using namespace gl::computing;

	/*
	 Represents a stored in device(GPU) memory model
	 buffer that allow get all information about model.
	 */
	class ModelBuffer {
	public:
		// Vertex collection.
		Vec3f *verts_;

		// Faces colection.
		Vec3i *faces_; // 2darray attention, this Vec3i means vertex/uv/normal

		// Collection of normals vectors.
		Vec3f *norms_;

		// Collection of uvs vectors.
		Vec2f *uv_;

		// Construct a model buffer from given CPU-stored model.
		ModelBuffer(Model *model);

		// Destructor: free all allocated GPU memory.
		~ModelBuffer();

		// Poiter to vertex number.
		int *nverts;

		// Pointer to faces number;
		int *nfaces;

		// Poiter to faces elements number.
		int *nfacesElem;

		// Gets a pointer to vertex number.
		__device__ int* getNVerts();

		// Gets a pointer to faces number.
		__device__ int* getNFaces();

		// Gets a pointer to faces element number.
		__device__ int* getNFacesElem();

		// Return a normal vector.
		__device__ Vec3f normal(int iface, int nthvert);

		// Return a normal vector from UV vector.
		__device__ Vec3f normal(Vec2f uv);

		// Return a vertex by index. 
		__device__ Vec3f vert(int i);

		// Return a vertex.
		__device__ Vec3f vert(int iface, int nthvert);

		// Return a UV vector.
		__device__ Vec2f uv(int iface, int nthvert);

		// Return a face.
		__device__ int face(int i, int idx);

		// Return a color for given UV vector.
		__device__ gl::Color::Device diffuse(Vec2f uvf);

		// Pointer to diffuse map texture buffer.
		TextureBuffer *diffuse_texture;

		// Pointer to nornal map texture buffer.
		TextureBuffer *normal_map_texture;
	};
}

#endif