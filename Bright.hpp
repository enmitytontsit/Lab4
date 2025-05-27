#pragma once

void runBrightnessIncrease();
void increaseBrightnessCPU(unsigned char* input, unsigned char* output, int width, int height, int delta);
__global__ void increaseBrightnessCUDA(unsigned char* input, unsigned char* output, int width, int height, int delta);
