
void random_image(unsigned char** out, int height, int width);

void load_image(unsigned char** out, int& height, int& width, char filename[]);

void save_image(unsigned char* in, int height, int width, char filename[]);

void add_noise(unsigned char* in, unsigned char** out, int height, int width, double noise_percent);
