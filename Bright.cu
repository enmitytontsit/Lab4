#include "brightness.hpp"
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

void increaseBrightnessCPU(unsigned char* input, unsigned char* output, int width, int height, int delta) {
    for (int i = 0; i < width * height; ++i) {
        int val = input[i] + delta;
        output[i] = (val > 255) ? 255 : val;
    }
}

__global__ void increaseBrightnessCUDA(unsigned char* input, unsigned char* output, int width, int height, int delta) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        int val = input[idx] + delta;
        output[idx] = (val > 255) ? 255 : val;
    }
}

void runBrightnessIncrease() {
    const int width = 1024, height = 1024, delta = 50;
    const int size = width * height;
    unsigned char* input = new unsigned char[size];
    unsigned char* output_cpu = new unsigned char[size];
    unsigned char* output_gpu = new unsigned char[size];

    for (int i = 0; i < size; ++i)
        input[i] = rand() % 256;

    auto start_cpu = std::chrono::high_resolution_clock::now();
    increaseBrightnessCPU(input, output_cpu, width, height, delta);
    auto end_cpu = std::chrono::high_resolution_clock::now();

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    increaseBrightnessCUDA<<<grid, block>>>(d_input, d_output, width, height, delta);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    cudaMemcpy(output_gpu, d_output, size, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < size; ++i) {
        if (output_cpu[i] != output_gpu[i]) {
            correct = false;
            break;
        }
    }

    std::chrono::duration<float> cpu_time = end_cpu - start_cpu;
    std::chrono::duration<float> gpu_time = end_gpu - start_gpu;

    std::cout << "CPU Time: " << cpu_time.count() << "s\n";
    std::cout << "GPU Time: " << gpu_time.count() << "s\n";
    std::cout << (correct ? "Brightness increase successful.\n" : "Mismatch in brightness.\n");

    delete[] input; delete[] output_cpu; delete[] output_gpu;
    cudaFree(d_input); cudaFree(d_output);
}
