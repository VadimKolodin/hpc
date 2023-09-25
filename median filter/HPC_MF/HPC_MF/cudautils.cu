#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdexcept>


void handle_cuda_result(cudaError_t cuerr, char msg[]) {
    if (cuerr != cudaSuccess) {
        fprintf(stderr, cudaGetErrorString(cuerr));
        fprintf(stderr, "\n");
       
    }
}

void upload_to_device(cudaArray** cudaArray, cudaChannelFormatDesc& channelDesc, int* in, int n, int m) {
    cudaError_t cuerr;
    cuerr = cudaMallocArray(cudaArray, &channelDesc, m, n);
    handle_cuda_result(cuerr, "Cannot allocate memory for array");
    cuerr = cudaMemcpyToArray(*cudaArray, 0, 0, in, sizeof(int) * n * m, cudaMemcpyHostToDevice);
    handle_cuda_result(cuerr, "Cannot copy to device");
}

void allocate_in_device(int** resultGpuPointer, int size) {
    int sizeInBytes = size * sizeof(int);
    // выделяем память
    cudaError_t cuerr = cudaMalloc((void**)resultGpuPointer, sizeInBytes);
    handle_cuda_result(cuerr, "Cannot allocate device array");
}

void download_from_device(int* gpuMatPointer, int* resultMat, int size) {
    int sizeInBytes = size * sizeof(int);
    // копируем массив
    cudaError_t cuerr = cudaMemcpy(resultMat, gpuMatPointer, sizeInBytes, cudaMemcpyDeviceToHost);
    handle_cuda_result(cuerr, "Cannot copy a array from device to host");
}