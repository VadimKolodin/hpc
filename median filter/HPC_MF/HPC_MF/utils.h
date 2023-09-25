
void random_image(int** out, int height, int width);

void load_image(int** out, int& height, int& width, char filename[]);

void save_image(int* in, int height, int width, char filename[]);

void add_noise(int* in, int** out, int height, int width, double noise_percent);
