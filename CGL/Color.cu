#include "Color.cuh"

// -------------------------------------------- Device ----------------------------------------------------------

// Pack RGBA color passed by channels to uint32 bit format.
// NOTICE: in bit format channels saving in order: ABGR.
__device__ Uint32 gl::Color::packRGBAToUint32(gl::Color::Channel r, gl::Color::Channel g,
	gl::Color::Channel b, gl::Color::Channel a)
{
	return (a << 24) | (b << 16) | (g << 8) | r;
}

// Pack from device RGBA color class to uint32 bit format.
// NOTICE: in bit format channels saving in order: ABGR.
__device__ Uint32 gl::Color::packColorToUint32(gl::Color::Device *color)
{
	return ((*color).alpha << 24) | ((*color).blue << 16) | ((*color).green << 8) | (*color).red;
}

// UnPack a unit32 bit format to device RGBA color class.
// NOTICE: in bit format channels saving in order: ABGR.
__device__ gl::Color::Device gl::Color::unpackUint32ToColor(Uint32 col)
{
	gl::Color::Device color;
	color.red = (*((Uint32*)col)) & 0xFF;
	color.green = (*((Uint32*)col) >> 8) & 0xFF;
	color.blue = (*((Uint32*)col) >> 16) & 0xFF;
	color.alpha = (*((Uint32*)col) >> 24) & 0xFF;
	return color;
}

// Multiplication operator overload (for change color intensivity).
__device__ gl::Color::Device gl::Color::Device::operator*(float intensity) const {
	gl::Color::Device res = *this;
	intensity = (intensity > 1.f ? 1.f : (intensity < 0.f ? 0.f : intensity));

	res.red = res.red * intensity;
	res.green = res.green * intensity;
	res.blue = res.blue * intensity;
	res.alpha = 255;
	return res;
}

// --------------------------------------------- Host -----------------------------------------------------------

// Pack from RGBA channels to uint32 bit format.
// NOTICE: in bit format channels saving in order: ABGR.
__host__ Uint32 gl::Color::getUint32FromRGBA(gl::Color::Channel r, gl::Color::Channel g,
	gl::Color::Channel b, gl::Color::Channel a)
{
	return (a << 24) | (b << 16) | (g << 8) | r;
}

// Pack from host RGBA color class to uint32 bit format.
// NOTICE: in bit format channels saving in order: ABGR.
__host__ Uint32 gl::Color::getUint32FromhColor(gl::Color::Host *color)
{
	return ((*color).alpha << 24) | ((*color).blue << 16) | ((*color).green << 8) | (*color).red;
}

// UnPack a unit32 bit format to host RGBA color.
// NOTICE: in bit format channels saving in order: ABGR.
__host__ gl::Color::Host gl::Color::gethColorFromUint32(Uint32 col)
{
	gl::Color::Host color;
	color.red = (*((Uint32*)col)) & 0xFF;
	color.green = (*((Uint32*)col) >> 8) & 0xFF;
	color.blue = (*((Uint32*)col) >> 16) & 0xFF;
	color.alpha = (*((Uint32*)col) >> 24) & 0xFF;
	return color;
}