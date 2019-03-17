#include "cuda_gl.cuh"

const int width = 800;
const int height = 800;

__device__ void viewport(int x, int y, int w, int h, Matrix &Viewport) {
	Viewport = Matrix::identity();
	Viewport[0][3] = x + w / 2.f;
	Viewport[1][3] = y + h / 2.f;
	Viewport[2][3] = 1.f;
	Viewport[0][0] = w / 2.f;
	Viewport[1][1] = h / 2.f;
	Viewport[2][2] = 0;
}

__device__ void projection(float coeff, Matrix &Projection) {
	Projection = Matrix::identity();
	Projection[3][2] = coeff;
}

__device__ void lookat(Vec3f eye, Vec3f center, Vec3f up, Matrix &ModelView) {
	Vec3f z = (eye - center).normalize();
	Vec3f x = cross(up, z).normalize();
	Vec3f y = cross(z, x).normalize();
	Matrix Minv = Matrix::identity();
	Matrix Tr = Matrix::identity();
	for (int i = 0; i < 3; i++) {
		Minv[0][i] = x[i];
		Minv[1][i] = y[i];
		Minv[2][i] = z[i];
		Tr[i][3] = -center[i];
	}
	ModelView = Minv * Tr;
}

__device__ Vec3f barycentric(Vec2f A, Vec2f B, Vec2f C, Vec2i P) {
	Vec3f s[2];
	for (int i = 2; i--; ) {
		s[i][0] = C[i] - A[i];
		s[i][1] = B[i] - A[i];
		s[i][2] = A[i] - P[i];
	}
	Vec3f u = cross(s[0], s[1]);
	if (abs(u[2]) > 1e-2) // dont forget that u[2] is integer. If it is zero then triangle ABC is degenerate
		return Vec3f(1.f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
	return Vec3f(-1, 1, 1); // in this case generate negative coordinates, it will be thrown away by the rasterizator
}

__device__ void triangle(mat<4, 3, float> &clipc, Shader &shader, void *pixels, int pinch, float *zbuffer, Matrix &Viewport) {
	mat<3, 4, float> pts = (Viewport*clipc).transpose(); // transposed to ease access to each of the points
    mat<3, 2, float> pts2;
	for (int i = 0; i < 3; i++) pts2[i] = proj<2>(pts[i] / pts[i][3]);

	Vec2f bboxmin(FLT_MAX, FLT_MAX);
	Vec2f bboxmax(-FLT_MAX, -FLT_MAX);
	Vec2f clamp(width - 1, height - 1); // with, height
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 2; j++) {
			bboxmin[j] = fmax(0.f, fmin(bboxmin[j], pts2[i][j]));
			bboxmax[j] = fmin(clamp[j], fmax(bboxmax[j], pts2[i][j]));
		}
	}
	Vec2i P;
	Color color;
	for (P.x = bboxmin.x; P.x <= bboxmax.x; P.x++) {
		for (P.y = bboxmin.y; P.y <= bboxmax.y; P.y++) {
			Vec3f bc_screen = barycentric(pts2[0], pts2[1], pts2[2], P);
			Vec3f bc_clip = Vec3f(bc_screen.x / pts[0][3], bc_screen.y / pts[1][3], bc_screen.z / pts[2][3]);
			bc_clip = bc_clip / (bc_clip.x + bc_clip.y + bc_clip.z);
			float frag_depth = clipc[2] * bc_clip;
			if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z<0 || zbuffer[P.x + P.y*width]>frag_depth) continue; // 800 - image with
			bool discard = true;//= shader.fragment(bc_clip, TGAColor());
			if (!discard) {
				zbuffer[P.x + P.y*width] = frag_depth;//image.width
				//setPixel(pixels, pinch, P.x, P.y, color);
			}
		}
	}
}

