#include "Drawing.cuh"

__device__ __forceinline__ void gl::draw::setPixel(void* pixels, int pinch, int x, int y, Color::Device *color)
{
	//printf("r=%d, g=%d, b=%d\n", (*color).red, (*color).green, (*color).blue);
	Uint8 *pixel = (Uint8*)pixels;
	pixel += ((800-y) * pinch) + (x * sizeof(Uint32));
	*((Uint32*)pixel) = packColorToUint32(color);//abgr
}

__device__ void swap(int &x, int &y)
{
	int tmp = x;
	x = y;
	y = tmp;
}

__device__ void swapVec2i(Vec2i &x, Vec2i &y)
{
	Vec2i tmp = x;
	x = y;
	y = tmp;
}

__device__ void swapVec3i(Vec3i &x, Vec3i &y)
{
	Vec3i tmp = x;
	x = y;
	y = tmp;
}

#define widthScreen 800


__device__ void gl::draw::triangle_s(mat<4, 3, float> *clipc, IShader *shader, void* pixels, int pinch, float *zbuffer, Matrix &Viewport, int ra) {
	mat<3, 4, float> pts = (Viewport*(*clipc)).transpose(); // transposed to ease access to each of the points
	mat<3, 2, float> pts2;
	for (int i = 0; i < 3; i++) pts2[i] = proj<2>(pts[i] / pts[i][3]);

	Vec2f bboxmin(FLT_MAX, FLT_MAX);
	Vec2f bboxmax(-FLT_MAX, -FLT_MAX);
	//Vec2f clamp(image.get_width() - 1, image.get_height() - 1);
	Vec2f clamp(799, 799); // 800-1 800-1
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 2; j++) {
			bboxmin[j] = fmaxf(0.f, fminf(bboxmin[j], pts2[i][j]));
			bboxmax[j] = fminf(clamp[j], fmaxf(bboxmax[j], pts2[i][j]));
		}
	}
	Vec2i P;
	Color::Device color;
	//printf("bboxmin.x=%f, bboxmax.x=%f\n", bboxmin.x, bboxmax.x);
	//printf("bboxmin.y=%f, bboxmax.y=%f\n", bboxmin.y, bboxmax.y);
	for (P.x = bboxmin.x; P.x <= bboxmax.x; P.x++) {
		for (P.y = bboxmin.y; P.y <= bboxmax.y; P.y++) {
		//	printf("P.x=%d, P.y=%d\n", P.x, P.y);
			Vec3f bc_screen = gl::camera::barycentric(pts2[0], pts2[1], pts2[2], P);
			Vec3f bc_clip = Vec3f(bc_screen.x / pts[0][3], bc_screen.y / pts[1][3], bc_screen.z / pts[2][3]);
			bc_clip = bc_clip / (bc_clip.x + bc_clip.y + bc_clip.z);
			float frag_depth = (*clipc)[2] * bc_clip;
	//		//if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z<0 || zbuffer[P.x + P.y*image.get_width()]>frag_depth) continue;
			if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z<0 || zbuffer[P.x + P.y * 800]>frag_depth) continue;
			bool discard = shader->fragment(bc_clip, color);
			if (!discard) {
				//printf("P.x=%d, P.y=%d\n", P.x, P.y);
	//			//zbuffer[P.x + P.y*image.get_width()] = frag_depth;
				zbuffer[P.x + P.y * 800] = frag_depth;
	//			//image.set(P.x, P.y, color);
				setPixel(pixels, pinch, P.x, P.y, &color);
			}
		}
	}
}

__device__ void gl::draw::triangleWihTex(Vec3i t0, Vec3i t1, Vec3i t2, Vec2i uv0, Vec2i uv1, Vec2i uv2,
	void* pixels, int pinch, float intensity, int *zbuffer, ModelBuffer *mb) {
	//printf("x0=%d, y0=%d || x1=%d, y1=%d || x2=%d, y2=%d\n", uv0.x, uv0.y, uv1.x, uv1.y, uv2.x, uv2.y);

	if (t0.y == t1.y && t0.y == t2.y) return; // i dont care about degenerate triangles
	if (t0.y > t1.y) { swapVec3i(t0, t1); swapVec2i(uv0, uv1); }
	if (t0.y > t2.y) { swapVec3i(t0, t2); swapVec2i(uv0, uv2); }
	if (t1.y > t2.y) { swapVec3i(t1, t2); swapVec2i(uv1, uv2); }

	int total_height = t2.y - t0.y;
	for (int i = 0; i < total_height; i++) {
		bool second_half = i > t1.y - t0.y || t1.y == t0.y;
		int segment_height = second_half ? t2.y - t1.y : t1.y - t0.y;
		float alpha = (float)i / total_height;
		float beta = (float)(i - (second_half ? t1.y - t0.y : 0)) / segment_height; // be careful: with above conditions no division by zero here
		Vec3i A = t0 + Vec3f(t2 - t0)*alpha;
		Vec3i B = second_half ? t1 + Vec3f(t2 - t1)*beta : t0 + Vec3f(t1 - t0)*beta;
		Vec2i uvA = uv0 + (uv2 - uv0)*alpha;
		Vec2i uvB = second_half ? uv1 + (uv2 - uv1)*beta : uv0 + (uv1 - uv0)*beta;
		if (A.x > B.x) { swapVec3i(A, B); swapVec2i(uvA, uvB); }
		
		for (int j = A.x; j <= B.x; j++) {
			float phi = B.x == A.x ? 1. : (float)(j - A.x) / (float)(B.x - A.x);
			Vec3i   P = Vec3f(A) + Vec3f(B - A)*phi;
			Vec2i uvP = uvA + (uvB - uvA)*phi;
			P.x = j; P.y = t0.y + i;//hack
			int idx = P.x + P.y*widthScreen;
			if (zbuffer[idx] < P.z) {
				zbuffer[idx] = P.z;
				Color::Device color = mb->diffuse(uvP);
				color.alpha = 255;
				color.red = color.red * intensity;
				color.green = color.green * intensity;
				color.blue = color.blue * intensity;
				
				setPixel(pixels, pinch, P.x, P.y, &color);
			}
		}
	}
}

__device__ void gl::draw::triangleZBuf(Vec3i t0, Vec3i t1, Vec3i t2, void* pixels, int pinch, Color::Device *col, float *zbuffer) {
	if (t0.y == t1.y && t0.y == t2.y) return; // i dont care about degenerate triangles
	if (t0.y > t1.y) swapVec3i(t0, t1);
	if (t0.y > t2.y) swapVec3i(t0, t2);
	if (t1.y > t2.y) swapVec3i(t1, t2);
	int total_height = t2.y - t0.y;
	for (int i = 0; i < total_height; i++) {
		bool second_half = i > t1.y - t0.y || t1.y == t0.y;
		int segment_height = second_half ? t2.y - t1.y : t1.y - t0.y;
		float alpha = (float)i / total_height;
		float beta = (float)(i - (second_half ? t1.y - t0.y : 0)) / segment_height; // be careful: with above conditions no division by zero here
		Vec3i A = t0 + Vec3f(t2 - t0)*alpha;
		Vec3i B = second_half ? t1 + Vec3f(t2 - t1)*beta : t0 + Vec3f(t1 - t0)*beta;
		if (A.x > B.x) swapVec3i(A, B);
		for (int j = A.x; j <= B.x; j++) {
			float phi = B.x == A.x ? 1. : (float)(j - A.x) / (float)(B.x - A.x);
			Vec3i P = Vec3f(A) + Vec3f(B - A)*phi;
			P.x = j; P.y = t0.y + i;//hack
			int idx = P.x + P.y*widthScreen;
			if (zbuffer[idx] < P.z) {
				zbuffer[idx] = P.z;
				setPixel(pixels, pinch, P.x, P.y, col);
			}
		}
	}
}

__device__ void gl::draw::triangle(Vec2i t0, Vec2i t1, Vec2i t2, void* pixels, int pinch, Color::Device *col) {
	if (t0.y == t1.y && t0.y == t2.y) return; // i dont care about degenerate triangles
	// sort the vertices, t0, t1, t2 lower-to-upper (bubblesort yay!)
	if (t0.y > t1.y) swapVec2i(t0, t1);
	if (t0.y > t2.y) swapVec2i(t0, t2);
	if (t1.y > t2.y) swapVec2i(t1, t2);
	int total_height = t2.y - t0.y;
	for (int i = 0; i < total_height; i++) {
		bool second_half = i > t1.y - t0.y || t1.y == t0.y;
		int segment_height = second_half ? t2.y - t1.y : t1.y - t0.y;
		float alpha = (float)i / total_height;
		float beta = (float)(i - (second_half ? t1.y - t0.y : 0)) / segment_height; // be careful: with above conditions no division by zero here
		Vec2i A = t0 + (t2 - t0)*alpha;
		Vec2i B = second_half ? t1 + (t2 - t1)*beta : t0 + (t1 - t0)*beta;
		if (A.x > B.x) swapVec2i(A, B);
		for (int j = A.x; j <= B.x; j++) {
			//image.set(j, t0.y + i, color); // attention, due to int casts t0.y+i != A.y
			setPixel(pixels, pinch, j, t0.y + i, col);
		}
	}
}

__device__ void gl::draw::line(int x0, int y0, int x1, int y1, void* pixels, int pinch, Color::Device *col) {
	bool steep = false;
	if (abs(x0 - x1) < abs(y0 - y1)) { // if the line is steep, we transpose the image
		swap(x0, y0);
		swap(x1, y1);
		steep = true;
	}
	if (x0 > x1) { // make it left-to-right
		swap(x0, x1);
		swap(y0, y1);
	}
	for (int x = x0; x <= x1; x++) {
		float t = (x - x0) / (float)(x1 - x0);
		int y = y0 * (1. - t) + y1 * t;
		if (steep) {
			setPixel(pixels, pinch, y, x, col); // if transposed, de-transpose
		}
		else {
			setPixel(pixels, pinch, x, y, col);
		}
	}
}


// ------------------------------- 2D drawing ------------------------------------------------------------------------

__host__ void gl::draw::SetPixel(Screen *screen, int x, int y, Color::Host color)
{
	screen->setPixel(x, y, getUint32FromhColor(&color));
}

__host__ void gl::draw::SetLine(Screen *screen, int x0, int y0, int x1, int y1, Color::Host color)
{
	bool steep = false;
	if (std::abs(x0 - x1) < std::abs(y0 - y1)) { // if the line is steep, we transpose the image
		std::swap(x0, y0);
		std::swap(x1, y1);
		steep = true;
	}
	if (x0 > x1) { // make it left-to-right
		std::swap(x0, x1);
		std::swap(y0, y1);
	}

	for (int x = x0; x <= x1; x++) {
		float t = (x - x0) / (float)(x1 - x0);
		int y = y0 * (1. - t) + y1 * t;
		if (steep) {
			SetPixel(screen, y, x, color); // if transposed, de-transpose
		}
		else {
			SetPixel(screen, x, y, color);
		}
	}
}

void gl::draw::SetTriangle(Screen *screen, int x0, int y0, int x1, int y1, int x2, int y2, Color::Host color)
{
	SetLine(screen, x0, y0, x1, y1, color);
	SetLine(screen, x1, y1, x2, y2, color);
	SetLine(screen, x2, y2, x0, y0, color);
}

void gl::draw::SetRectangle(Screen *screen, int x0, int y0, int x1, int y1, int x2, int y2, int x3, int y3, Color::Host color)
{
	SetLine(screen, x0, y0, x1, y1, color);
	SetLine(screen, x1, y1, x2, y2, color);
	SetLine(screen, x2, y2, x3, y3, color);
	SetLine(screen, x3, y3, x0, y0, color);
}

void gl::draw::SetCircle(Screen *screen, int xc, int yc, int radius, Color::Host color)
{
	int x = 0, y = radius;
	int d = 3 - 2 * radius;
	SetPixel(screen, xc + x, yc + y, color);
	SetPixel(screen, xc - x, yc + y, color);
	SetPixel(screen, xc + x, yc - y, color);
	SetPixel(screen, xc - x, yc - y, color);
	SetPixel(screen, xc + y, yc + x, color);
	SetPixel(screen, xc - y, yc + x, color);
	SetPixel(screen, xc + y, yc - x, color);
	SetPixel(screen, xc - y, yc - x, color);
	while (y >= x)
	{
		// for each pixel we will 
		// draw all eight pixels 

		x++;

		// check for decision parameter 
		// and correspondingly  
		// update d, x, y 
		if (d > 0)
		{
			y--;
			d = d + 4 * (x - y) + 10;
		}
		else
		{
			d = d + 4 * x + 6;
		}
		SetPixel(screen, xc + x, yc + y, color);
		SetPixel(screen, xc - x, yc + y, color);
		SetPixel(screen, xc + x, yc - y, color);
		SetPixel(screen, xc - x, yc - y, color);
		SetPixel(screen, xc + y, yc + x, color);
		SetPixel(screen, xc - y, yc + x, color);
		SetPixel(screen, xc + y, yc - x, color);
		SetPixel(screen, xc - y, yc - x, color);
	}
}

void gl::draw::SetPolygon(Screen *screen, int* coordPair, int coordsSize, Color::Host color)
{
	for (int i = 0; i < coordsSize - 2; i += 2)
	{
		SetLine(screen, coordPair[i], coordPair[i + 1], coordPair[i + 2], coordPair[i + 3], color);
	}
}

void gl::draw::SetImage(Screen *screen, TGAImage *image)
{
	image->scale(screen->GetWidth(), screen->GetHeight());
	for (int x = 0; x < image->width; x++)
	{
		for (int y = 0; y < image->height; y++)
		{
			TGAColor tgaColor = image->get(x, y);
			SetPixel(screen, x, y, gl::Color::Host(tgaColor[2], tgaColor[1], tgaColor[0]));
		}
	}
}