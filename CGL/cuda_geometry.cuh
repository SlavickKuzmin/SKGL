#ifndef _CUDA_GEOMETRY_CUH_
#define _CUDA_GEOMETRY_CUH_

#include <cassert>
#include "cuda_runtime_api.h"

template<size_t DimCols, size_t DimRows, typename T> class mat;

template <size_t DIM, typename T> struct vec {
	__device__  vec() { for (size_t i = DIM; i--; data_[i] = T()); }
	__device__ T& operator[](const size_t i) { assert(i < DIM); return data_[i]; }
	__device__ const T& operator[](const size_t i) const { assert(i < DIM); return data_[i]; }
private:
	 T data_[DIM];
};

/////////////////////////////////////////////////////////////////////////////////

template <typename T> struct vec<2, T> {
	__device__ vec() : x(T()), y(T()) {}
	__device__ vec(T X, T Y) : x(X), y(Y) {}
	template <class U> __device__ vec<2, T>(const vec<2, U> &v);
	__device__ T& operator[](const size_t i) { assert(i < 2); return i <= 0 ? x : y; }
	__device__ const T& operator[](const size_t i) const { assert(i < 2); return i <= 0 ? x : y; }

	T x, y;
};

/////////////////////////////////////////////////////////////////////////////////

template <typename T> struct vec<3, T> {
	__device__ vec() : x(T()), y(T()), z(T()) {}
	__device__ vec(T X, T Y, T Z) : x(X), y(Y), z(Z) {}
	template <class U> __device__ vec<3, T>(const vec<3, U> &v);
	__device__ T& operator[](const size_t i) { assert(i < 3); return i <= 0 ? x : (1 == i ? y : z); }
	__device__ const T& operator[](const size_t i) const { assert(i < 3); return i <= 0 ? x : (1 == i ? y : z); }
	__device__ float norm() { return std::sqrt(x*x + y * y + z * z); }
	__device__ vec<3, T> & normalize(T l = 1) { *this = (*this)*(l / norm()); return *this; }

	T x, y, z;
};

/////////////////////////////////////////////////////////////////////////////////

template<size_t DIM, typename T> __device__ T operator*(const vec<DIM, T>& lhs, const vec<DIM, T>& rhs) {
	T ret = T();
	for (size_t i = DIM; i--; ret += lhs[i] * rhs[i]);
	return ret;
}


template<size_t DIM, typename T> __device__ vec<DIM, T> operator+(vec<DIM, T> lhs, const vec<DIM, T>& rhs) {
	for (size_t i = DIM; i--; lhs[i] += rhs[i]);
	return lhs;
}

template<size_t DIM, typename T> __device__ vec<DIM, T> operator-(vec<DIM, T> lhs, const vec<DIM, T>& rhs) {
	for (size_t i = DIM; i--; lhs[i] -= rhs[i]);
	return lhs;
}

template<size_t DIM, typename T, typename U> __device__ vec<DIM, T> operator*(vec<DIM, T> lhs, const U& rhs) {
	for (size_t i = DIM; i--; lhs[i] *= rhs);
	return lhs;
}

template<size_t DIM, typename T, typename U> __device__ vec<DIM, T> operator/(vec<DIM, T> lhs, const U& rhs) {
	for (size_t i = DIM; i--; lhs[i] /= rhs);
	return lhs;
}

template<size_t LEN, size_t DIM, typename T> __device__ vec<LEN, T> embed(const vec<DIM, T> &v, T fill = 1) {
	vec<LEN, T> ret;
	for (size_t i = LEN; i--; ret[i] = (i < DIM ? v[i] : fill));
	return ret;
}

template<size_t LEN, size_t DIM, typename T> __device__ vec<LEN, T> proj(const vec<DIM, T> &v) {
	vec<LEN, T> ret;
	for (size_t i = LEN; i--; ret[i] = v[i]);
	return ret;
}

template <typename T> __device__ vec<3, T> cross(vec<3, T> v1, vec<3, T> v2) {
	return vec<3, T>(v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x);
}

/////////////////////////////////////////////////////////////////////////////////

template<size_t DIM, typename T> struct dt {
	__device__ static T det(const mat<DIM, DIM, T>& src) {
		T ret = 0;
		for (size_t i = DIM; i--; ret += src[0][i] * src.cofactor(0, i));
		return ret;
	}
};

template<typename T> struct dt<1, T> {
	__device__ static T det(const mat<1, 1, T>& src) {
		return src[0][0];
	}
};

/////////////////////////////////////////////////////////////////////////////////

template<size_t DimRows, size_t DimCols, typename T> class mat {
	 vec<DimCols, T> rows[DimRows];
public:
	 __device__ mat() {}

	__device__ vec<DimCols, T>& operator[] (const size_t idx) {
		assert(idx < DimRows);
		return rows[idx];
	}

	__device__ const vec<DimCols, T>& operator[] (const size_t idx) const {
		assert(idx < DimRows);
		return rows[idx];
	}

	__device__ vec<DimRows, T> col(const size_t idx) const {
		assert(idx < DimCols);
		vec<DimRows, T> ret;
		for (size_t i = DimRows; i--; ret[i] = rows[i][idx]);
		return ret;
	}

	__device__ void set_col(size_t idx, vec<DimRows, T> v) {
		assert(idx < DimCols);
		for (size_t i = DimRows; i--; rows[i][idx] = v[i]);
	}

	__device__ static mat<DimRows, DimCols, T> identity() {
		mat<DimRows, DimCols, T> ret;
		for (size_t i = DimRows; i--; )
			for (size_t j = DimCols; j--; ret[i][j] = (i == j));
		return ret;
	}

	__device__ T det() const {
		return dt<DimCols, T>::det(*this);
	}

	__device__ mat<DimRows - 1, DimCols - 1, T> get_minor(size_t row, size_t col) const {
		mat<DimRows - 1, DimCols - 1, T> ret;
		for (size_t i = DimRows - 1; i--; )
			for (size_t j = DimCols - 1; j--; ret[i][j] = rows[i < row ? i : i + 1][j < col ? j : j + 1]);
		return ret;
	}

	__device__ T cofactor(size_t row, size_t col) const {
		return get_minor(row, col).det()*((row + col) % 2 ? -1 : 1);
	}

	__device__ mat<DimRows, DimCols, T> adjugate() const {
		mat<DimRows, DimCols, T> ret;
		for (size_t i = DimRows; i--; )
			for (size_t j = DimCols; j--; ret[i][j] = cofactor(i, j));
		return ret;
	}

	__device__ mat<DimRows, DimCols, T> invert_transpose() {
		mat<DimRows, DimCols, T> ret = adjugate();
		T tmp = ret[0] * rows[0];
		return ret / tmp;
	}

	__device__ mat<DimRows, DimCols, T> invert() {
		return invert_transpose().transpose();
	}

	__device__ mat<DimCols, DimRows, T> transpose() {
		mat<DimCols, DimRows, T> ret;
		for (size_t i = DimCols; i--; ret[i] = this->col(i));
		return ret;
	}
};

/////////////////////////////////////////////////////////////////////////////////

template<size_t DimRows, size_t DimCols, typename T> __device__ vec<DimRows, T> operator*(const mat<DimRows, DimCols, T>& lhs, const vec<DimCols, T>& rhs) {
	vec<DimRows, T> ret;
	for (size_t i = DimRows; i--; ret[i] = lhs[i] * rhs);
	return ret;
}

template<size_t R1, size_t C1, size_t C2, typename T> __device__ mat<R1, C2, T> operator*(const mat<R1, C1, T>& lhs, const mat<C1, C2, T>& rhs) {
	mat<R1, C2, T> result;
	for (size_t i = R1; i--; )
		for (size_t j = C2; j--; result[i][j] = lhs[i] * rhs.col(j));
	return result;
}

template<size_t DimRows, size_t DimCols, typename T> __device__ mat<DimCols, DimRows, T> operator/(mat<DimRows, DimCols, T> lhs, const T& rhs) {
	for (size_t i = DimRows; i--; lhs[i] = lhs[i] / rhs);
	return lhs;
}

/////////////////////////////////////////////////////////////////////////////////

typedef vec<2, float> Vec2f;
typedef vec<2, int>   Vec2i;
typedef vec<3, float> Vec3f;
typedef vec<3, int>   Vec3i;
typedef vec<4, float> Vec4f;
typedef mat<4, 4, float> Matrix;

#endif // !_CUDA_GEOMETRY_CUH_
