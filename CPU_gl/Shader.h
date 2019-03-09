#pragma once
#include "model.h"
#include "geometry.h"
#include "our_gl.h"
#include <vector>
#include <limits>
#include <algorithm>

class Shader : public IShader {
public:
	mat<2, 3, float> varying_uv;  // triangle uv coordinates, written by the vertex shader, read by the fragment shader
	mat<4, 3, float> varying_tri; // triangle coordinates (clip coordinates), written by VS, read by FS
	mat<3, 3, float> varying_nrm; // normal per vertex to be interpolated by FS
	mat<3, 3, float> ndc_tri;     // triangle in normalized device coordinates

	virtual Vec4f vertex(int iface, int nthvert);
	virtual bool fragment(Vec3f bar, TGAColor &color);

	Shader(Model *model, Vec3f *light_dir);

private:
	Model *model;
	Vec3f light_dir;
};