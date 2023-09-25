#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdexcept>
#include <iostream>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#include "cudautils.cuh"
#include "utils.h"


using namespace std;

texture<int, 2, cudaReadModeElementType> tex;

__device__ void bubble_sort(int* array, int n) {
    int i, j, temp;
    for (i = 0; i < n - 1; i++) {
        for (j = 0; j < n - i - 1; j++) {
            if (array[j] > array[j + 1]) {
                temp = array[j];
                array[j] = array[j + 1];
                array[j + 1] = temp;
            }
        }
    }
}

/*
 Фильтрация медианным фильтром
*/
__global__ void kernel(int* out, int n, int m, int k, int sharedArrayOffsetScale) {
    extern __shared__ int sharedArray[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int shared_array_offset = sharedArrayOffsetScale * (threadIdx.x * blockDim.y + threadIdx.y);

    if (i >= n || j >= m) {
        return;
    }

    int mini = max(0, i - k);
    int minj = max(0, j - k);
    int maxi = min(n-1, i + k);
    int maxj = min(m-1, j + k);

    int cn = (maxi - mini) + 1;
    int cm = (maxj - minj) + 1;
   
    for (int ci = 0; ci < cn; ++ci) {
        for (int cj = 0; cj < cm; ++cj) {
            // складываем найденное в буфер
            sharedArray[shared_array_offset + ci * cm + cj] = tex2D(tex, minj + cj, mini + ci);
        }
    }
   
    // сортируем для получения медианы
    bubble_sort(sharedArray + shared_array_offset, cn * cm);
    out[i * m + j] = sharedArray[shared_array_offset + cn * cm / 2];

}

void count_cuda_dims(dim3& blocksPerGrid, dim3& threadsPerBlock, int n, int m) {
    // максимум потоков в блоке - 1024
    int xthreadsPerBlock = n < 32 ? n : 32;
    int ythreadsPerBlock = m < 32 ? m : 32;
    // дальше считаем кличесто блоков. С учетом ограничений по заданию не упремся в лимит точно (там что-то типа 65К)
    int xblocksPerGrid = ceil((float)n / xthreadsPerBlock);
    int yblocksPerGrid = ceil((float)m / ythreadsPerBlock);
    //printf("grid(%d, %d) block(%d,%d)\n", xblocksPerGrid, yblocksPerGrid, xthreadsPerBlock, ythreadsPerBlock);
    // z не используем
    blocksPerGrid = dim3(xblocksPerGrid, yblocksPerGrid, 1);
    threadsPerBlock = dim3(xthreadsPerBlock, ythreadsPerBlock, 1);

}

/*
  Фильтрация иображения in(n, m) с помощью медианного фильтра
*/
double filter_gpu(int* in, int** out, int n, int m, int kernel_size) {
    cudaError_t cuerr;
    // Выделение памяти на устройстве
    int* outGpuPointer;
    cudaArray* inCudaArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
    int k = (kernel_size - 1) / 2;

    allocate_in_device(&outGpuPointer, n * m);
    upload_to_device(&inCudaArray, channelDesc, in, n, m);
  
    cuerr = cudaBindTextureToArray(&tex, inCudaArray, &channelDesc);
    handle_cuda_result(cuerr, "Cannot create CUDA start event");

    // Создание обработчиков событий
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cuerr = cudaEventCreate(&start);
    handle_cuda_result(cuerr, "Cannot create CUDA start event");

    cuerr = cudaEventCreate(&stop);
    handle_cuda_result(cuerr, "Cannot create CUDA stop event");

    cuerr = cudaEventRecord(start, 0);
    handle_cuda_result(cuerr, "Cannot record CUDA start event");

    dim3 blocksPerGrid, threadsPerBlock;
    count_cuda_dims(blocksPerGrid, threadsPerBlock, n, m);

    int sharedArrayOffsetScale = kernel_size * kernel_size;
    int bufSize = sharedArrayOffsetScale * threadsPerBlock.x * threadsPerBlock.y;
    
    // Запуск ядра
    kernel<<<blocksPerGrid, threadsPerBlock, bufSize * sizeof(int)>>>(outGpuPointer, n, m, k, sharedArrayOffsetScale);
    handle_cuda_result(cudaGetLastError(), "Cannot launch CUDA kernel");

    // Синхронизация устройств
    cuerr = cudaDeviceSynchronize();
    handle_cuda_result(cudaGetLastError(), "Cannot synchronize CUDA kernel");

    // Установка точки окончания
    cuerr = cudaEventRecord(stop, 0);
    handle_cuda_result(cuerr, "Cannot record CUDA stop event");

    // Копирование результата на хост
    *out = new int[n * m];
    download_from_device(outGpuPointer, *out, n*m);

    cuerr = cudaEventElapsedTime(&gpuTime, start, stop);
    handle_cuda_result(cuerr, "Cannot get elapsed time");
    double time = gpuTime / 1000;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(inCudaArray);
    cudaFree(outGpuPointer);
    return time;
}


int comp(const int* a, const int* b) {
    return *a - *b;
}

double filter_cpu(int* in, int** out, int n, int m, int kernel_size) {
    int* output = new int[n * m];
    int kernel = (kernel_size - 1) / 2;

    double time = clock();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            // опеределяем границы "свертки", чтобы было читаемее
            int mini = max(0, i - kernel);
            int minj = max(0, j - kernel);
            int maxi = min(n-1, i + kernel);
            int maxj = min(m-1, j + kernel);
            
            int cn = (maxi - mini) + 1;
            int cm = (maxj - minj) + 1;
            int* temp_array = new int[cn*cm];

            for (int ci = 0; ci < cn; ++ci) {
                for (int cj = 0; cj < cm; ++cj) {
                    // складываем найденное в буфер
                    temp_array[ci * cm + cj] = in[(mini + ci) * m + (minj + cj)];
                }
            }

            // сортируем для получения медианы
            qsort(temp_array, cn * cm, sizeof(int), (int(*) (const void*, const void*)) comp);
            output[i * m + j] = temp_array[cn * cm / 2];
            delete[] temp_array;
        }
    }
    time = clock() - time;
    time /= CLOCKS_PER_SEC;

    *out = output;

    return time;
}

void test_real_image() {
    int height, width;
    int* image;
    int filter_size = 3;
    load_image(&image, height, width, "mushroom.bmp");

    int* image_with_noise;
    add_noise(image, &image_with_noise, height, width, 0.1);
    save_image(image_with_noise, height, width, "image_with_noise.bmp");
    delete[] image;

    int* image_filtered_cpu;
    float time_cpu = filter_cpu(image_with_noise, &image_filtered_cpu, height, width, filter_size);
    save_image(image_filtered_cpu, height, width, "image_filtered_cpu.bmp");
    delete[] image_filtered_cpu;
    printf("time cpu: %f\n", time_cpu);

    int* image_filtered_gpu;
    float time_gpu = filter_gpu(image_with_noise, &image_filtered_gpu, height, width, filter_size);
    save_image(image_filtered_gpu, height, width, "image_filtered_gpu.bmp");
    delete[] image_filtered_gpu;
    printf("time gpu: %f\n", time_gpu);

    delete[] image_with_noise;
}

void test_fake_image() {
    int height = 2160, width = 3840;
    int* image;
    int filter_size = 3;
    random_image(&image, height, width);
    save_image(image, height, width, "random.bmp");

    int* image_with_noise;
    add_noise(image, &image_with_noise, height, width, 0.1);
    save_image(image_with_noise, height, width, "random_with_noise.bmp");

    int* image_filtered_cpu;
    float time_cpu = filter_cpu(image_with_noise, &image_filtered_cpu, height, width, filter_size);
    save_image(image_filtered_cpu, height, width, "random_filtered_cpu.bmp");
    delete[] image_filtered_cpu;
    printf("time cpu: %f\n", time_cpu);

    int* image_filtered_gpu;
    float time_gpu = filter_gpu(image_with_noise, &image_filtered_gpu, height, width, filter_size);
    save_image(image_filtered_gpu, height, width, "random_filtered_gpu.bmp");
    delete[] image_filtered_gpu;
    printf("time gpu: %f\n", time_gpu);

    delete[] image_with_noise;
}

int main(int argc, char* argv[]) {
    test_fake_image();
    //test_real_image();
    return 0;
}