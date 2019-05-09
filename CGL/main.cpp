#include <algorithm>
#include "model.h"
#include <SDL.h>
#undef main

#include "RenderOnGPU.cuh"

#include "Screen.h"

#include <Windows.h>

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_IMPLEMENTATION
#define NK_SDL_GL2_IMPLEMENTATION
#include <nuklear.h>
#include "nuklear_sdl_gl2.h"


#ifdef INCLUDE_ALL
#define INCLUDE_STYLE
#define INCLUDE_CALCULATOR
#define INCLUDE_OVERVIEW
#define INCLUDE_NODE_EDITOR
#endif

#ifdef INCLUDE_STYLE
#include "../style.c"
#endif
#ifdef INCLUDE_CALCULATOR
#include "../calculator.c"
#endif
#ifdef INCLUDE_OVERVIEW
#include "../overview.c"
#endif
#ifdef INCLUDE_NODE_EDITOR
#include "../node_editor.c"
#endif

const int width = 800;
const int height = 800;
void runRender();
void runGPURender();

int main() {
	//runRender();
	runGPURender();
	getchar();
	return EXIT_SUCCESS;
}

void runGPURender()
{
	// SDL main loop
	SDL_Event event;
	SDL_Window *window;
	SDL_GLContext glContext;

	SDL_Init(SDL_INIT_VIDEO);
	window = SDL_CreateWindow("GPU render",
		SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
		width, height, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);

	glContext = SDL_GL_CreateContext(window);

	// my inits
	gl::Screen *screen = new gl::Screen(width, height);
	//for(int i = 0; i < height; i++)
	//		screen->setPixel(i, height-i, 0xFFFF0000);

	Model *model = new Model("E:\\Diplom\\SDL\\CGL\\obj\\diablo3_pose\\diablo3_pose.obj");
	TGAImage *background = new TGAImage();
	background->read_tga_file("E:\\Diplom\\sky.tga");
	//Model *model = new Model("E:\\Diplom\\SDL\\CGL\\obj\\african_head\\african_head.obj");
	//Model *model = new Model("E:\\3d\\stature\\statue.obj");
	gl::RenderOnGPU *render = new gl::RenderOnGPU(model, screen);

	bool quit = false;

	//Event handler
	SDL_Event e;

	Vec3f light_dir(1, 1, 1);
	Vec3f       eye(1.f, 1.f, 1.f);
	Vec3f    center(0, 0, 0);
	Vec3f        up(0, 1, 0);

	float step = 0.3f;

	char title[5];
	gl::RenderMode mode = gl::RenderMode::Shaders;
	bool UseBackground = false;

	// GUI
	struct nk_context *ctx;
	struct nk_colorf bg;

	/* GUI */
	ctx = nk_sdl_init(window);
	/* Load Fonts: if none of these are loaded a default font will be used  */
	/* Load Cursor: if you uncomment cursor loading please hide the cursor */
	{struct nk_font_atlas *atlas;
	nk_sdl_font_stash_begin(&atlas);
	/*struct nk_font *droid = nk_font_atlas_add_from_file(atlas, "../../../extra_font/DroidSans.ttf", 14, 0);*/
	/*struct nk_font *roboto = nk_font_atlas_add_from_file(atlas, "../../../extra_font/Roboto-Regular.ttf", 16, 0);*/
	/*struct nk_font *future = nk_font_atlas_add_from_file(atlas, "../../../extra_font/kenvector_future_thin.ttf", 13, 0);*/
	/*struct nk_font *clean = nk_font_atlas_add_from_file(atlas, "../../../extra_font/ProggyClean.ttf", 12, 0);*/
	/*struct nk_font *tiny = nk_font_atlas_add_from_file(atlas, "../../../extra_font/ProggyTiny.ttf", 10, 0);*/
	/*struct nk_font *cousine = nk_font_atlas_add_from_file(atlas, "../../../extra_font/Cousine-Regular.ttf", 13, 0);*/
	nk_sdl_font_stash_end();
	/*nk_style_load_all_cursors(ctx, atlas->cursors);*/
	/*nk_style_set_font(ctx, &roboto->handle)*/; }

#ifdef INCLUDE_STYLE
	/*set_style(ctx, THEME_WHITE);*/
	/*set_style(ctx, THEME_RED);*/
	/*set_style(ctx, THEME_BLUE);*/
	/*set_style(ctx, THEME_DARK);*/
#endif

	bg.r = 0.10f, bg.g = 0.18f, bg.b = 0.24f, bg.a = 1.0f;

	//--------------
	glOrtho(0.0f, width, height, 0.0f, 0.0f, -1.0f);
	/*glMatrixMode(GL_MODELVIEW);
	glShadeModel(GL_SMOOTH);
	glClearColor(0.0f, 40.0f / 256.0f, 100.0f / 256.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearDepth(1.0f);
	*/
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);



	//While application is running
	while (!quit)
	{
		nk_input_begin(ctx);
		//Handle events on queue
		while (SDL_PollEvent(&e) != 0)
		{
			nk_sdl_handle_event(&e);
			//User requests quit
			if (e.type == SDL_QUIT)
			{
				quit = true;
			}
			//User presses a key
			else if (e.type == SDL_KEYDOWN)
			{
				//Select surfaces based on key press
				switch (e.key.keysym.sym)
				{
				case SDLK_UP:
					printf("up=%f\n", eye.y);
					eye.y += step;
					break;
				case SDLK_DOWN:
					printf("down=%f\n", eye.y);
					eye.y += -step;
					break;
				case SDLK_LEFT:
					printf("left=%f\n", eye.x);
					eye.x += step;
					break;
				case SDLK_RIGHT:
					printf("left=%f\n", eye.x);
					eye.x += -step;
					break;
				case SDLK_z:
					printf("z+=%f\n", eye.z);
					eye.z += step;
					break;
				case SDLK_x:
					printf("z-=%f\n", eye.z);
					eye.z += -step;
					break;
				case SDLK_1:
					mode = gl::RenderMode::Shaders;
					break;
				case SDLK_2:
					mode = gl::RenderMode::Filled;
					break;
				case SDLK_3:
					mode = gl::RenderMode::Wire;
					break;
				case SDLK_4:
					mode = gl::RenderMode::ShadersWithWire;
					break;
				case SDLK_q:
					UseBackground = !UseBackground;
					break;
				default:
					break;
				}
			}
		}
		nk_input_end(ctx);
		
		/* window flags */
		static int titlebar = nk_true;
		static int border = nk_true;
		static int resize = nk_true;
		static int movable = nk_true;
		static int no_scrollbar = nk_false;
		static int scale_left = nk_false;
		static nk_flags window_flags = 0;
		static int minimizable = nk_true;

		/* popups */
		static enum nk_style_header_align header_align = NK_HEADER_RIGHT;
		static int show_app_about = nk_false;

		/* window flags */
		window_flags = 0;
		ctx->style.window.header.align = header_align;
		if (border) window_flags |= NK_WINDOW_BORDER;
		if (resize) window_flags |= NK_WINDOW_SCALABLE;
		if (movable) window_flags |= NK_WINDOW_MOVABLE;
		if (no_scrollbar) window_flags |= NK_WINDOW_NO_SCROLLBAR;
		if (scale_left) window_flags |= NK_WINDOW_SCALE_LEFT;
		if (minimizable) window_flags |= NK_WINDOW_MINIMIZABLE;

		/* GUI */
		if(nk_begin(ctx, "Settings", nk_rect(50, 50, 400, 600),
			NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE |
			NK_WINDOW_MINIMIZABLE | NK_WINDOW_TITLE))
		{
				
			static const float ratio[] = { 150, 150 };
			static char text[9][64];
			static int text_len[9];

			nk_layout_row_dynamic(ctx, 30, 1);
			//nk_layout_row(ctx, NK_STATIC, 25, 2, ratio);
			nk_label(ctx, "Enter path to OBJ model:", NK_TEXT_LEFT);

			nk_edit_string(ctx, NK_EDIT_SIMPLE, text[0], &text_len[0], 64, nk_filter_default);

			if (nk_button_label(ctx, "Load"))
			{
				printf("start model loading\n");
				//delete model;
				//delete render; ??? why not work TODO !!!!!
				model = new Model((char*)text);
				render = new gl::RenderOnGPU(model, screen);
				printf("model loading - OK\n");
			}

				/* window flags */
				if (nk_tree_push(ctx, NK_TREE_TAB, "Background", NK_MINIMIZED)) {
					static int inactive = 1;
					nk_layout_row_dynamic(ctx, 30, 1);
					nk_label(ctx, "Background color:", NK_TEXT_LEFT);
					/* complex color combobox */
					if (nk_combo_begin_color(ctx, nk_rgb_cf(bg), nk_vec2(200, 400))) {
						enum color_mode { COL_RGB, COL_HSV };
						static int col_mode = COL_RGB;
#ifndef DEMO_DO_NOT_USE_COLOR_PICKER
						nk_layout_row_dynamic(ctx, 120, 1);
						bg = nk_color_picker(ctx, bg, NK_RGBA);
#endif

						nk_layout_row_dynamic(ctx, 25, 1);
						bg.r = nk_propertyf(ctx, "#R:", 0, bg.r, 1.0f, 0.01f, 0.005f);
						bg.g = nk_propertyf(ctx, "#G:", 0, bg.g, 1.0f, 0.01f, 0.005f);
						bg.b = nk_propertyf(ctx, "#B:", 0, bg.b, 1.0f, 0.01f, 0.005f);
						bg.a = nk_propertyf(ctx, "#A:", 0, bg.a, 1.0f, 0.01f, 0.005f);
						
						nk_combo_end(ctx);
					}


					nk_checkbox_label(ctx, "Use TGA image as background", &inactive);

					//nk_layout_row_static(ctx, 30, 80, 1);
					if (inactive) {
						struct nk_style_button button;
						button = ctx->style.button;
						ctx->style.button.normal = nk_style_item_color(nk_rgb(40, 40, 40));
						ctx->style.button.hover = nk_style_item_color(nk_rgb(40, 40, 40));
						ctx->style.button.active = nk_style_item_color(nk_rgb(40, 40, 40));
						ctx->style.button.border_color = nk_rgb(60, 60, 60);
						ctx->style.button.text_background = nk_rgb(60, 60, 60);
						ctx->style.button.text_normal = nk_rgb(60, 60, 60);
						ctx->style.button.text_hover = nk_rgb(60, 60, 60);
						ctx->style.button.text_active = nk_rgb(60, 60, 60);
						if (nk_button_label(ctx, "Set image"))
						{
							
						}
						ctx->style.button = button;
						UseBackground = false;
					}
					else
					{
						UseBackground = true;
						static const float ratio[] = { 150, 150 };
						static char text[9][64];
						static int text_len[9];

						nk_layout_row_dynamic(ctx, 30, 1);
						//nk_layout_row(ctx, NK_STATIC, 25, 2, ratio);
						nk_label(ctx, "Default:", NK_TEXT_LEFT);

						nk_edit_string(ctx, NK_EDIT_SIMPLE, text[0], &text_len[0], 64, nk_filter_default);

						if (nk_button_label(ctx, "Set image"))
						{
							background->read_tga_file((char*)text);
						}
					}
					nk_tree_pop(ctx);
				}

				
				if (nk_tree_push(ctx, NK_TREE_TAB, "Render mode", NK_MINIMIZED))
				{
					nk_layout_row_dynamic(ctx, 30, 1);
					nk_label(ctx, "Choose a render mode:", NK_TEXT_LEFT);

					mode = nk_option_label(ctx, "Wire", mode == gl::RenderMode::Wire) ? gl::RenderMode::Wire : mode;
					mode = nk_option_label(ctx, "Filled", mode == gl::RenderMode::Filled) ? gl::RenderMode::Filled : mode;
					mode = nk_option_label(ctx, "Shaders", mode == gl::RenderMode::Shaders) ? gl::RenderMode::Shaders : mode;
					mode = nk_option_label(ctx, "Shaders with wire", mode == gl::RenderMode::ShadersWithWire) ? gl::RenderMode::ShadersWithWire : mode;

					nk_tree_pop(ctx);
				}

				if (nk_tree_push(ctx, NK_TREE_TAB, "Transformations:", NK_MINIMIZED))
				{
					if (nk_tree_push(ctx, NK_TREE_NODE, "Eye:", NK_MINIMIZED))
					{
						nk_layout_row_dynamic(ctx, 30, 1);
						nk_label(ctx, "Rotate a object:", NK_TEXT_LEFT);
						nk_layout_space_begin(ctx, NK_STATIC, 60, 4);
						nk_layout_space_push(ctx, nk_rect(100, 0, 100, 30));
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_TRIANGLE_UP, "up", NK_TEXT_CENTERED))
						{
							eye.y += step;
						}
						nk_layout_space_push(ctx, nk_rect(0, 15, 100, 30));
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_TRIANGLE_LEFT, "left", NK_TEXT_LEFT))
						{
							eye.x += step;
						}
						nk_layout_space_push(ctx, nk_rect(200, 15, 100, 30));
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_TRIANGLE_RIGHT, "right", NK_TEXT_CENTERED))
						{
							eye.x += -step;
						}
						nk_layout_space_push(ctx, nk_rect(100, 30, 100, 30));
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_TRIANGLE_DOWN, "down", NK_TEXT_CENTERED))
						{
							eye.y += -step;
						}
						nk_layout_space_end(ctx);

						// Z direction

						nk_layout_row_dynamic(ctx, 30, 1);
						nk_label(ctx, "Moving by Z-coordinate:", NK_TEXT_LEFT);
						nk_layout_row_template_begin(ctx, 30);
						nk_layout_row_template_push_dynamic(ctx);
						nk_layout_row_template_push_variable(ctx, 80);
						nk_layout_row_template_push_static(ctx, 80);
						nk_layout_row_template_end(ctx);
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_CIRCLE_SOLID, "Deeply", NK_TEXT_CENTERED))
						{
							eye.z += step;
						}
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_CIRCLE_OUTLINE, "Closely", NK_TEXT_CENTERED))
						{
							eye.z += -step;
						}
						nk_tree_pop(ctx);
					}
					
					if (nk_tree_push(ctx, NK_TREE_NODE, "Light direction:", NK_MINIMIZED))
					{
						// light dir
						nk_layout_row_dynamic(ctx, 30, 1);
						nk_label(ctx, "Change light position:", NK_TEXT_LEFT);
						nk_layout_space_begin(ctx, NK_STATIC, 60, 4);
						nk_layout_space_push(ctx, nk_rect(100, 0, 100, 30));
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_TRIANGLE_UP, "up", NK_TEXT_CENTERED))
						{
							light_dir.y += step;
						}
						nk_layout_space_push(ctx, nk_rect(0, 15, 100, 30));
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_TRIANGLE_LEFT, "left", NK_TEXT_LEFT))
						{
							light_dir.x += step;
						}
						nk_layout_space_push(ctx, nk_rect(200, 15, 100, 30));
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_TRIANGLE_RIGHT, "right", NK_TEXT_CENTERED))
						{
							light_dir.x += -step;
						}
						nk_layout_space_push(ctx, nk_rect(100, 30, 100, 30));
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_TRIANGLE_DOWN, "down", NK_TEXT_CENTERED))
						{
							light_dir.y += -step;
						}
						nk_layout_space_end(ctx);

						nk_layout_row_dynamic(ctx, 30, 1);
						nk_label(ctx, "Moving Z-coordinate:", NK_TEXT_LEFT);
						nk_layout_row_template_begin(ctx, 30);
						nk_layout_row_template_push_dynamic(ctx);
						nk_layout_row_template_push_variable(ctx, 80);
						nk_layout_row_template_push_static(ctx, 80);
						nk_layout_row_template_end(ctx);
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_CIRCLE_SOLID, "deeply", NK_TEXT_CENTERED))
						{
							light_dir.z += step;
						}
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_CIRCLE_OUTLINE, "closely", NK_TEXT_CENTERED))
						{
							light_dir.z += -step;
						}
						nk_tree_pop(ctx);
					}
					
					if (nk_tree_push(ctx, NK_TREE_NODE, "Center:", NK_MINIMIZED))
					{
						//
						// center
						nk_layout_row_dynamic(ctx, 30, 1);
						nk_label(ctx, "Change center position:", NK_TEXT_LEFT);
						nk_layout_space_begin(ctx, NK_STATIC, 60, 4);
						nk_layout_space_push(ctx, nk_rect(100, 0, 100, 30));
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_TRIANGLE_UP, "up", NK_TEXT_CENTERED))
						{
							center.y += step;
						}
						nk_layout_space_push(ctx, nk_rect(0, 15, 100, 30));
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_TRIANGLE_LEFT, "left", NK_TEXT_LEFT))
						{
							center.x += step;
						}
						nk_layout_space_push(ctx, nk_rect(200, 15, 100, 30));
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_TRIANGLE_RIGHT, "right", NK_TEXT_CENTERED))
						{
							center.x += -step;
						}
						nk_layout_space_push(ctx, nk_rect(100, 30, 100, 30));
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_TRIANGLE_DOWN, "down", NK_TEXT_CENTERED))
						{
							center.y += -step;
						}
						nk_layout_space_end(ctx);

						nk_layout_row_dynamic(ctx, 30, 1);
						nk_label(ctx, "Move Z-coordinate:", NK_TEXT_LEFT);
						nk_layout_row_template_begin(ctx, 30);
						nk_layout_row_template_push_dynamic(ctx);
						nk_layout_row_template_push_variable(ctx, 80);
						nk_layout_row_template_push_static(ctx, 80);
						nk_layout_row_template_end(ctx);
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_CIRCLE_SOLID, "deeply", NK_TEXT_CENTERED))
						{
							center.z += step;
						}
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_CIRCLE_OUTLINE, "closely", NK_TEXT_CENTERED))
						{
							center.z += -step;
						}
						nk_tree_pop(ctx);
					}
					
					if (nk_tree_push(ctx, NK_TREE_NODE, "Up vector:", NK_MINIMIZED))
					{
						//
						// up
						nk_layout_row_dynamic(ctx, 30, 1);
						nk_label(ctx, "Change up vector:", NK_TEXT_LEFT);
						nk_layout_space_begin(ctx, NK_STATIC, 60, 4);
						nk_layout_space_push(ctx, nk_rect(100, 0, 100, 30));
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_TRIANGLE_UP, "up", NK_TEXT_CENTERED))
						{
							up.y += step;
						}
						nk_layout_space_push(ctx, nk_rect(0, 15, 100, 30));
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_TRIANGLE_LEFT, "left", NK_TEXT_LEFT))
						{
							up.x += step;
						}
						nk_layout_space_push(ctx, nk_rect(200, 15, 100, 30));
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_TRIANGLE_RIGHT, "right", NK_TEXT_CENTERED))
						{
							up.x += -step;
						}
						nk_layout_space_push(ctx, nk_rect(100, 30, 100, 30));
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_TRIANGLE_DOWN, "down", NK_TEXT_CENTERED))
						{
							up.y += -step;
						}
						nk_layout_space_end(ctx);

						nk_layout_row_dynamic(ctx, 30, 1);
						nk_label(ctx, "Move Z-coordinate:", NK_TEXT_LEFT);
						nk_layout_row_template_begin(ctx, 30);
						nk_layout_row_template_push_dynamic(ctx);
						nk_layout_row_template_push_variable(ctx, 80);
						nk_layout_row_template_push_static(ctx, 80);
						nk_layout_row_template_end(ctx);
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_CIRCLE_SOLID, "deeply", NK_TEXT_CENTERED))
						{
							up.z += step;
						}
						nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
						if (nk_button_symbol_label(ctx, NK_SYMBOL_CIRCLE_OUTLINE, "closely", NK_TEXT_CENTERED))
						{
							up.z += -step;
						}
						nk_tree_pop(ctx);
					}
					nk_tree_pop(ctx);
				}
		}

		nk_end(ctx);

		if (UseBackground)
		{
			gl::draw::SetImage(screen, background);
		}
		else
		{
			screen->ClearScreen(gl::Color::Host(bg.r, bg.g, bg.b, bg.a));
		}
		render->refresh(light_dir, eye, center, up, mode);

		GLuint TextureID = screen->GetTextureScreen();

		glClear(GL_COLOR_BUFFER_BIT);
		glClearColor(bg.r, bg.g, bg.b, bg.a);

		// For Ortho mode, of course
		int X = 0;
		int Y = 0;

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, TextureID);
		glBegin(GL_QUADS);
		glTexCoord2f(0, 0); glVertex3f(X, Y, 0);
		glTexCoord2f(1, 0); glVertex3f(X + width, Y, 0);
		glTexCoord2f(1, 1); glVertex3f(X + width, Y + height, 0);
		glTexCoord2f(0, 1); glVertex3f(X, Y + height, 0);
		glEnd();
		glDisable(GL_TEXTURE_2D);


		nk_sdl_render(NK_ANTI_ALIASING_ON);
		SDL_GL_SwapWindow(window);

		
		glDeleteTextures(1, &TextureID);
		// FPS count
		int fps = 1.0f / render->GetRenderFrameTime();
		sprintf(title, "FPS: %d", fps);
		SDL_SetWindowTitle(window, title);
	}

	delete screen;
	delete model;
	delete background;

	delete render;
	SDL_GL_DeleteContext(glContext);
	nk_sdl_shutdown();
	SDL_DestroyWindow(window);
	SDL_Quit();
}