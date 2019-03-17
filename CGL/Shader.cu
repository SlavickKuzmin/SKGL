#include "Shader.cuh"

__device__ Vec4f Shader::vertex(int iface, int nthvert, Matrix ModelView, Matrix Projection, Vec2f uv, Vec3f normal, Vec3f vert) {
	varying_uv.set_col(nthvert, uv);// model->uv(iface, nthvert));
	varying_nrm.set_col(nthvert, proj<3>((Projection*ModelView).invert_transpose()*embed<4>(normal, 0.f)));//model->normal(iface, nthvert)
	Vec4f gl_Vertex = Projection * ModelView*embed<4>(vert);//model->vert(iface, nthvert));
	varying_tri.set_col(nthvert, gl_Vertex);
	ndc_tri.set_col(nthvert, proj<3>(gl_Vertex / gl_Vertex[3]));
	return gl_Vertex;
}

__device__ bool Shader::fragment(Vec3f bar, TGAColor &color, Vec3f normal, TGAColor diffuse) {
	Vec3f bn = (varying_nrm*bar).normalize();
	Vec2f uv = varying_uv * bar;

	mat<3, 3, float> A;
	A[0] = ndc_tri.col(1) - ndc_tri.col(0);
	A[1] = ndc_tri.col(2) - ndc_tri.col(0);
	A[2] = bn;

	mat<3, 3, float> AI = A.invert();

	Vec3f i = AI * Vec3f(varying_uv[0][1] - varying_uv[0][0], varying_uv[0][2] - varying_uv[0][0], 0);
	Vec3f j = AI * Vec3f(varying_uv[1][1] - varying_uv[1][0], varying_uv[1][2] - varying_uv[1][0], 0);

	mat<3, 3, float> B;
	B.set_col(0, i.normalize());
	B.set_col(1, j.normalize());
	B.set_col(2, bn);

	Vec3f n = (B*normal).normalize();//model->normal(uv)

	float diff = fmax(0.f, n*light_dir);
	//color = diffuse * diff;//model->diffuse(uv)*diff;

	return false;
}

Shader::Shader(Vec3f *light_dir)
{
	// make cuda copy
	this->light_dir = *light_dir;
}