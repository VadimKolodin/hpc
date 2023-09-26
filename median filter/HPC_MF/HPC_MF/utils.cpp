#include <stdio.h>
#include <stdlib.h>
#include <EasyBMP.h>

void random_image(unsigned char** out, int height, int width) {
	srand(time(NULL));
	unsigned char* output = new unsigned char[width * height];
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			output[i * width + j] = i*j/4;
			output[i * width + j] = output[i * width + j] % 255;
			output[i * width + j] = output[i * width + j] < 0 ? 0 : output[i * width + j];
		}
	}
	*out = output;
}

void load_image(unsigned char** out, int& height, int& width, char filename[]) {
	BMP image;
	image.ReadFromFile(filename);
	width = image.TellWidth();
	height = image.TellHeight();
	unsigned char* output = new unsigned char[width * height];
	for (int i = 0; i < image.TellHeight(); i++) {
		for (int j = 0; j < image.TellWidth(); j++) {
			output[i * width + j] = (unsigned char)floor(0.299 * image(j, i)->Red +
				0.587 * image(j, i)->Green +
				0.114 * image(j, i)->Blue);
		}
	}
	*out = output;
}

void save_image(unsigned char* in, int height, int width, char filename[]) {
	BMP image;
	image.SetSize(width, height);
	width = image.TellWidth();
	height = image.TellHeight();
	for (int i = 0; i < image.TellHeight(); i++) {
		for (int j = 0; j < image.TellWidth(); j++) {
			ebmpBYTE TempBYTE = (ebmpBYTE)in[i * width + j];
			image(j, i)->Red = TempBYTE;
			image(j, i)->Green = TempBYTE;
			image(j, i)->Blue = TempBYTE;
		}
	}

	if (image.TellBitDepth() < 16) {
		CreateGrayscaleColorTable(image);
	}

	image.WriteToFile(filename);
}


void add_noise(unsigned char* in, unsigned char** out, int height, int width, double noise_percent) {
	srand(time(NULL));
	unsigned char* output = new unsigned char[width * height];
	int noise_pixels_count = noise_percent * width * height;
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			output[i * width + j] = in[i * width + j];
		}
	}
	for (int k = 0; k < noise_pixels_count; ++k) {
		int i = rand() % height;
		int j = rand() % width;
		bool is_white = rand() % 2;
		output[i * width + j] = 255 * is_white;
	}
	*out = output;
}
