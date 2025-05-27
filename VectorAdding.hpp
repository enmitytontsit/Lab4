#pragma once

void runVectorAddition();
void vectorAddCPU(const float* A, const float* B, float* C, int N);
__global__ void vectorAddCUDA(const float* A, const float* B, float* C, int N);
