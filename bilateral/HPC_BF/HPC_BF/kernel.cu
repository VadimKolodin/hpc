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

texture<unsigned char, 2, cudaReadModeElementType> tex;

__constant__ const float COEF1 = 1 / (2 * 15 * 15 * 3.1415);
__constant__ const float DOUBLE_SIGMA_SQUARE = 2 * 15 * 15;

__device__ float gauss_cu(float x) {
    return COEF1 * exp(-(x * x) / DOUBLE_SIGMA_SQUARE);
}


__global__ void kernel(unsigned char* out, int n, int m, int k) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= m) {
        return;
    }

    int mini = max(0, i - k);
    int minj = max(0, j - k);
    int maxi = min(n-1, i + k);
    int maxj = min(m-1, j + k);

    int cn = (maxi - mini) + 1;
    int cm = (maxj - minj) + 1;

    float coef = 0;
    float value = 0;
    float g, r, f, gr;
    
    float center = tex2D(tex, j, i) / 255.0;

    for (int ci = 0; ci < cn; ++ci) {
        for (int cj = 0; cj < cm; ++cj) {
            f = tex2D(tex, minj + cj, mini + ci) / 255.0;
            g = gauss_cu(f - center);
            // тут приведение типов к double, тк в CUDA 11.2 есть перегрузка только pow(double, double)
            r = gauss_cu(pow(double(mini + ci - i), 2.0) + pow(double(minj + cj - j), 2.0));
           
            gr = g * r;

            value += f * gr;
            coef += gr;

        }
    }

    out[i * m + j] = ((int)(255 * value / coef)) % 255;

}

void count_cuda_dims(dim3& blocksPerGrid, dim3& threadsPerBlock, int n, int m) {
    // максимум потоков в блоке - 1024, но тогда может не хватить памяти на буфер, поэтому ограничиваемся 512
    int xthreadsPerBlock = n < 16 ? n : 16;
    int ythreadsPerBlock = m < 16 ? m : 16;
    // дальше считаем кличесто блоков. С учетом ограничений по заданию не упремся в лимит точно (там что-то типа 65К)
    int xblocksPerGrid = ceil((float)n / xthreadsPerBlock);
    int yblocksPerGrid = ceil((float)m / ythreadsPerBlock);
    //printf("grid(%d, %d) block(%d,%d)\n", xblocksPerGrid, yblocksPerGrid, xthreadsPerBlock, ythreadsPerBlock);
    // z не используем
    blocksPerGrid = dim3(xblocksPerGrid, yblocksPerGrid, 1);
    threadsPerBlock = dim3(xthreadsPerBlock, ythreadsPerBlock, 1);

}

/*
  Фильтрация иображения in(n, m) с помощью фильтра Гаусса
*/
double filter_gpu(unsigned char* in, unsigned char** out, int n, int m, int kernel_size) {
    cudaError_t cuerr;
    // Выделение памяти на устройстве
    unsigned char* outGpuPointer;
    cudaArray* inCudaArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    int k = (kernel_size - 1) / 2;

    allocate_in_device(&outGpuPointer, n * m);
    upload_to_device(&inCudaArray, channelDesc, in, n, m);
  
    cuerr = cudaBindTextureToArray(&tex, inCudaArray, &channelDesc);
    handle_cuda_result(cuerr, "Cannot bind texture");

    // Создание обработчиков событий
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cuerr = cudaEventCreate(&start);
    handle_cuda_result(cuerr, "Cannot create CUDA start event");

    cuerr = cudaEventCreate(&stop);
    handle_cuda_result(cuerr, "Cannot create CUDA stop event");

    dim3 blocksPerGrid, threadsPerBlock;
    count_cuda_dims(blocksPerGrid, threadsPerBlock, n, m);

    cuerr = cudaEventRecord(start, 0);
    handle_cuda_result(cuerr, "Cannot record CUDA start event");
    
    // Запуск ядра
    kernel << <blocksPerGrid, threadsPerBlock >> > (outGpuPointer, n, m, k);
    handle_cuda_result(cudaGetLastError(), "Cannot launch CUDA kernel");

    // Синхронизация устройств
    cuerr = cudaDeviceSynchronize();
    handle_cuda_result(cudaGetLastError(), "Cannot synchronize CUDA kernel");

    // Установка точки окончания
    cuerr = cudaEventRecord(stop, 0);
    handle_cuda_result(cuerr, "Cannot record CUDA stop event");

    // Копирование результата на хост
    *out = new unsigned char[n * m];
    download_from_device(outGpuPointer, *out, n*m);

    cuerr = cudaEventElapsedTime(&gpuTime, start, stop);
    handle_cuda_result(cuerr, "Cannot get elapsed time");
    double time = gpuTime / 1000;

    cuerr = cudaUnbindTexture(&tex);
    handle_cuda_result(cuerr, "Cannot unbind texture");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(inCudaArray);
    cudaFree(outGpuPointer);
    return time;
}


__host__ float gauss(float x) {
    return COEF1 * exp(-(x * x) / DOUBLE_SIGMA_SQUARE);
}

double filter_cpu(unsigned char* in, unsigned char** out, int n, int m, int kernel_size) {
    unsigned char* output = new unsigned char[n * m];
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

            float k = 0;
            float value = 0;
            int index0 = i * m + j;
            for (int ci = 0; ci < cn; ++ci) {
                for (int cj = 0; cj < cm; ++cj) {
                    int index = (mini + ci) * m + (minj + cj);
                    float g = gauss((in[index]-in[index0])/255.0);
                    float r = gauss(pow(mini + ci-i, 2)+pow(minj + cj-j,2));
                    float f = in[index] / 255.0;
                   
                    value += f * r * g;
                    k += g * r;
                }
            }
           
            output[i * m + j] = ((int)(255*value / k)) % 255;
        }
    }
    time = clock() - time;
    time /= CLOCKS_PER_SEC;

    *out = output;

    return time;
}

void test_real_image(int filter_size) {
    int height, width;
    unsigned char* image;
 
    load_image(&image, height, width, "mushroom.bmp");

    unsigned char* image_filtered_cpu;
    float time_cpu = filter_cpu(image, &image_filtered_cpu, height, width, filter_size);
    save_image(image_filtered_cpu, height, width, "image_filtered_cpu.bmp");
    delete[] image_filtered_cpu;
    printf("time cpu: %f\n", time_cpu);

    unsigned char* image_filtered_gpu;
    float time_gpu = filter_gpu(image, &image_filtered_gpu, height, width, filter_size);
    save_image(image_filtered_gpu, height, width, "image_filtered_gpu.bmp");
    delete[] image_filtered_gpu;
    printf("time gpu: %f\n", time_gpu);

    delete[] image;
}

void test_fake_image(int h, int w, int filter_size) {
    unsigned char* image;
    
    random_image(&image, h, w);
    save_image(image, h, w, "random.bmp");

    unsigned char* image_filtered_cpu;
    float time_cpu = filter_cpu(image, &image_filtered_cpu, h, w, filter_size);
    save_image(image_filtered_cpu, h, w, "random_filtered_cpu.bmp");
    delete[] image_filtered_cpu;
    printf("time cpu: %f\n", time_cpu);

    unsigned char* image_filtered_gpu;
    float time_gpu = filter_gpu(image, &image_filtered_gpu, h, w, filter_size);
    save_image(image_filtered_gpu, h, w, "random_filtered_gpu.bmp");
    delete[] image_filtered_gpu;
    printf("time gpu: %f\n", time_gpu);

    delete[] image;
}

int main(int argc, char* argv[]) {
    test_fake_image(1080, 1920, 9);
    //test_real_image(7);
    return 0;
}