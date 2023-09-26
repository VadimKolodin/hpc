#include "cuda_runtime.h"

void handle_cuda_result(cudaError_t cuerr, char msg[]);

/*
Загрузка данных на хост-машину с GPU
 @ resultGpuPointer - указатель на память GPU
 @ gpuMatPointer - указатель на массив с развернутой матрицей, куда будет загружена матрица (память должная быть выделена)
 @ size - размер развернутой матрицы
*/
void download_from_device(unsigned char* gpuMatPointer, unsigned char* resultMat, int size);

/*
 Загрузка данных на GPU
 @ cudaArray - перезаписывается, указатель на указатель на память GPU (чтобы значение указателя сохранялось)
 @ channelDesc - сведения о типе данных текстуры
 @ in - указатель на массив с развернутой матрицей
 @ n, m - размеры развернутой матрицы
*/
void upload_to_device(cudaArray** cudaArray, cudaChannelFormatDesc& channelDesc, unsigned char* in, int n, int m);

/*
 Выделение памяти на GPU
 @ resultGpuPointer - перезаписывается, указатель на указатель на память GPU (чтобы значение указателя сохранялось)
 @ size - размер выделяемой памяти
*/
void allocate_in_device(unsigned char** resultGpuPointer, int size);