#ifndef _CUDA_UTILS_H_
#define _CUDA_UTILS_H_

#include <cuda.h>
#include "cuda_runtime_api.h"
#include <stdio.h>

namespace gl
{
	class CudaUtils
	{
	public:
		CudaUtils();
		static void cudasafe(int error, char* message, char* file, int line);
		static void PrintDeviceInfo();
		~CudaUtils();
	};
}

#endif // _CUDA_UTILS_H_