#ifndef _GPU_KERNELS_H_
#define _GPU_KERNELS_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef unsigned char uchar;
typedef unsigned int  uint;

int gpuWarmup(int t);
void callGrayscaleKernel(uchar3 *bgr_ptr, float *gray_ptr, int rows, int cols);
void callROIKernel(float *src, int src_rows, int src_cols, float *dst, int x, int y, int width, int height);
void callTemplateMatchKernel(float *tm, int tm_rows, int tm_cols, float *src, int src_rows, int src_cols, float *output);
void callMinLocKernel(float *src, int rows, int cols, uint * min_loc);

#endif
