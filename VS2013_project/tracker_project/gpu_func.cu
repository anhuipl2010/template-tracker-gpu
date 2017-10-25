#include "gpu_func.h"
#include <iostream>
#include <cmath>

/*
	Inline function to check if image index is in bounds
*/
static __device__ __forceinline__ bool in_img(int x, int y, int rows, int cols)
{
	return x >= 0 && x < cols && y >= 0 && y < rows;
}

__global__ void device_add_one(int* d_result, int t)
{
	*d_result = t + 1;
}

/*
	Dummy function which is used to warm up GPU
*/
int gpuWarmup(int t)
{
	int result;
	int *d_result;

	cudaMalloc((void **)&d_result, 1 * sizeof(int));
	device_add_one << <1, 1 >> >(d_result, t);
	cudaMemcpy(&result, d_result, 1 * sizeof(int), cudaMemcpyDeviceToHost);

	return result;
}


/*
	Converts BGR image to grayscale
*/
template<int px_per_thread>
__global__ void bgr_to_grayscale(uchar3 *bgr_ptr, float *gray_ptr, int rows, int cols)
{
	// get global index within image
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = px_per_thread * (blockIdx.y * blockDim.y + threadIdx.y);

	// loop over the number of pixels each thread is handling
	for (size_t i = 0; i < px_per_thread; ++i)
	{
		// get BGR pixel values
		uchar3 p;
		if (in_img(x, y + i, rows, cols))
			p = bgr_ptr[(y + i) * cols + x];
		else
			return;

		// calculate grayscale value
		float g = 0.298839f*(float)p.z + 0.586811f*(float)p.y + 0.114350f*(float)p.x;

		// set grayscale value in image
		if (in_img(x, y + i, rows, cols))
			gray_ptr[(y + i) * cols + x] = (g >= 255.f ? 255.f : g);
	}
}

void callGrayscaleKernel(uchar3 *bgr_ptr, float *gray_ptr, int rows, int cols) {
	// define num pixels each thread operates on
	const int px_per_thread = 4;
	
	// define block and grid sizes
	dim3 block_size(32, 8);
	dim3 grid_size(0, 0);
	grid_size.x = (cols + block_size.x - 1) / block_size.x;
	grid_size.y = (rows + px_per_thread * block_size.y - 1) / (px_per_thread * block_size.y);

	// call grayscale conversion kernel
	bgr_to_grayscale<px_per_thread> << <grid_size, block_size >> >(bgr_ptr, gray_ptr, rows, cols);
	return;
}


template<int px_per_thread>
__global__ void copy_roi(float *src, int src_rows, int src_cols,
						 float *dst, int x, int y, int width, int height)
{
	// get global index within image
	const int x_dst = blockIdx.x * blockDim.x + threadIdx.x;
	const int y_dst = px_per_thread * (blockIdx.y * blockDim.y + threadIdx.y);

	// loop over the number of pixels each thread is handling
	for (size_t i = 0; i < px_per_thread; i++)
	{
		// make sure we're in bounds of both src and dst images
		if (in_img(x + x_dst, y + y_dst + i, src_rows, src_cols) && in_img(x_dst, y_dst, height, width))
			dst[(y_dst + i) * width + x_dst] = src[(y + y_dst + i) * src_cols + x_dst + x];
	}
	return;
}

/*
	Call kernel to copy section of source image into new image
*/
void callROIKernel(float *src, int src_rows, int src_cols,
				   float *dst, int x, int y, int width, int height)
{
	// define num pixels each thread operates on
	const int px_per_thread = 4;

	// define block and grid sizes
	dim3 block_size(32, 8);
	dim3 grid_size(0, 0);
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + px_per_thread * block_size.y - 1) / (px_per_thread * block_size.y);

	// call grayscale conversion kernel
	copy_roi<px_per_thread> << <grid_size, block_size >> >(src, src_rows, src_cols, dst, x, y, width, height);
	return;
}



__global__ void sqrd_diff_normed(float *tm, int tm_rows, int tm_cols,
								 float *src, int src_rows, int src_cols,
								 float *output)
{
	// get global index within output
	const int x = blockIdx.x * blockDim.x + threadIdx.x;	// global column
	const int y = blockIdx.y * blockDim.y + threadIdx.y;	// global row

	// ensure in bounds
	uint output_rows = src_rows - tm_rows + 1;
	uint output_cols = src_cols - tm_cols + 1;
	if (!in_img(x, y, output_rows, output_cols))	return;

	float sqrd_diff = 0;
	float tm_sqrd_sum = 0;
	float src_sqrd_sum = 0;

	// iterate over all template image pixels and correspondences
	for (uint i = 0; i < tm_rows; i++) {
		for (uint j = 0; j < tm_cols; j++) {

			float tm_val = tm[i * tm_cols + j];
			float src_val = src[(y + i) * src_cols + x + j];

			// increment squared difference
			float diff = tm_val - src_val;
			sqrd_diff += diff * diff;

			// increment squared sum
			tm_sqrd_sum += tm_val * tm_val;
			src_sqrd_sum += src_val * src_val;

		}
	}

	output[y * output_cols + x] = 255.f * sqrd_diff / sqrt(tm_sqrd_sum * src_sqrd_sum);
	return;
}


/*
	Perform template matching by calling kernel
*/
void callTemplateMatchKernel(float *tm, int tm_rows, int tm_cols,		// template info
							 float *src, int src_rows, int src_cols,	// source info
							 float *output)								// output
{
	uint output_rows = src_rows - tm_rows + 1;
	uint output_cols = src_cols - tm_cols + 1;

	// each thread computes one pixel in the result
	dim3 block_size(32, 8);
	dim3 grid_size(0, 0);
	grid_size.x = (output_cols + block_size.x - 1) / block_size.x;
	grid_size.y = (output_rows + block_size.y - 1) / block_size.y;

	sqrd_diff_normed << <grid_size, block_size >> >(tm, tm_rows, tm_cols, src, src_rows, src_cols, output);
	return;
}


__global__ void find_min_loc(float *src, int rows, int cols, uint * min_loc) {
	float min_val = 256.f;
	for (uint i = 0; i < rows; i++) {
		for (uint j = 0; j < cols; j++) {
			float im_val = src[i * cols + j];
			if (im_val < min_val) {
				min_val = im_val;
				min_loc[0] = j;	// col
				min_loc[1] = i;	// row
			}
		}
	}
	return;
}

/*
	Slowly and naively determines location of minimum value in device image
*/
void callMinLocKernel(float *src, int rows, int cols, uint * min_loc) {
	find_min_loc << <1, 1 >> >(src, rows, cols, min_loc);
	return;
}
