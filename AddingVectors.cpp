#include "vector_add.hpp"
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>

void vectorAddCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i)
        C[i] = A[i] + B[i];
}

__global__ void vectorAddCUDA(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

void runVectorAddition() {
    const int N = 1000000;
    size_t size = N * sizeof(float);
    float *A = new float[N], *B = new float[N], *C_cpu = new float[N], *C_gpu = new float[N];

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(N - i);
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(A, B, C_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    auto start_gpu = std::chrono::high_resolution_clock::now();
    vectorAddCUDA<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    cudaMemcpy(C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(C_cpu[i] - C_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    std::chrono::duration<float> cpu_time = end_cpu - start_cpu;
    std::chrono::duration<float> gpu_time = end_gpu - start_gpu;

    std::cout << "CPU Time: " << cpu_time.count() << "s\n";
    std::cout << "GPU Time: " << gpu_time.count() << "s\n";
    std::cout << (correct ? "Results are correct.\n" : "Mismatch in results!\n");

    delete[] A; delete[] B; delete[] C_cpu; delete[] C_gpu;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
