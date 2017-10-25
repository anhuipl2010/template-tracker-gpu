#include "DeviceImage.h"


/*
	Uploads BGR color image from host to device
*/
void DeviceImage::upload(cv::Mat& host_img) {
	// ensure image is color
	if (host_img.type() != CV_8UC3)
	{
		std::cerr << "ERR: Can only upload 3-channel color images" << std::endl;
		return;
	}

	// get pointer to host image data
	uchar3 *host_ptr = (uchar3 *)host_img.ptr<uchar>(0);

	// allocate GPU memory
	nRows = host_img.rows;
	nCols = host_img.cols;
	const uint num_pixels = nRows * nCols;
	cudaMalloc((void**)&bgr, num_pixels * sizeof(uchar3));
	if (bgr == 0)
	{
		std::cerr << "ERR: Failed to allocate device memory for BGR image" << std::endl;
		return;
	}

	// copy host data to device
	cudaMemcpy(&bgr[0], &host_ptr[0], num_pixels * sizeof(uchar3), cudaMemcpyHostToDevice);
	return;
}


/*
	Downloads an image from device to host.
	User may specify which type of image (GRAY or COLOR).
*/
cv::Mat DeviceImage::download(ImageType imType = ImageType::GRAY) {
	// copy image from GPU
	const uint num_pixels = nRows * nCols;
	cv::Mat output;
	switch (imType)
	{
	case ImageType::GRAY:
	{
		// ensure gray image exists on device
		if (gray == 0) {
			std::cerr << "ERR: No gray image in GPU memory" << std::endl;
			return output;
		}
		// copy gray image from device to host
		float *host_img = (float *)malloc(num_pixels * sizeof(float));
		cudaMemcpy(&host_img[0], &gray[0], num_pixels * sizeof(float), cudaMemcpyDeviceToHost);
		output = cv::Mat(nRows, nCols, CV_32FC1, (void*)host_img);
		break;
	}
	case ImageType::COLOR:
	{
		// ensure color image exists on device
		if (bgr == 0) {
			std::cerr << "ERR: No color image in GPU memory" << std::endl;
			return output;
		}
		// copy color image from device to host
		uchar3 *host_img = (uchar3 *)malloc(num_pixels * sizeof(uchar3));
		cudaMemcpy(&host_img[0], bgr, num_pixels * sizeof(uchar3), cudaMemcpyDeviceToHost);
		output = cv::Mat(nRows, nCols, CV_8UC3, (void*)host_img);
		break;
	}
	}
	return output;
}


/*
	Converts color image on device to gray.
	Calls grayscale conversion kernel.
*/
void DeviceImage::convertToGrayscale() {
	// ensure color image exists on device
	if (bgr == 0) {
		std::cerr << "ERR: No color image loaded on device" << std::endl;
		return;
	}

	// allocate memory for gray image
	cudaMalloc((void**)&gray, nRows * nCols * sizeof(float));
	if (gray == 0)
	{
		std::cerr << "ERR: Failed to allocate memory for gray image" << std::endl;
		return;
	}

	// call conversion kernel
	callGrayscaleKernel(bgr, gray, nRows, nCols);
	return;
}


/*
	Copies a region from device image
	NOTE: top_left = (col, row)
*/
DeviceImage DeviceImage::getROI(cv::Point top_left, int width, int height) {
	// ensure gray image exists
	if (gray == 0)
	{
		std::cerr << "ERR: Can't get ROI since no gray image found" << std::endl;
		return DeviceImage();
	}

	// allocate device memory
	uint num_pixels = width * height;
	float * output_ptr = 0;
	cudaMalloc((void**)&output_ptr, num_pixels * sizeof(float));
	if (output_ptr == 0)
	{
		std::cerr << "ERR: Failed to allocate ROI data on device" << std::endl;
	}

	// call region extraction kernel
	callROIKernel(gray, nRows, nCols, output_ptr, top_left.x, top_left.y, width, height);
	DeviceImage result(height, width, 0, output_ptr);
	result.lockDeviceData();	// ensure device data isn't freed after we leave scope
	return result;
}

/*
	Performs template matching on GPU.
*/
DeviceImage DeviceImage::templateMatch(DeviceImage& tm) {
	// allocate memory on device
	uint output_rows = nRows - tm.rows() + 1;
	uint output_cols = nCols - tm.cols() + 1;
	float * output_ptr = 0;
	cudaMalloc((void**)&output_ptr,  output_rows * output_cols * sizeof(float));
	if (output_ptr == 0)
	{
		std::cerr << "ERR: Failed to allocate data for template matching" << std::endl;
	}

	// perform template matching
	callTemplateMatchKernel(tm.grayPtr(), tm.rows(), tm.cols(), gray, nRows, nCols, output_ptr);
	DeviceImage result(output_rows, output_cols, 0, output_ptr);
	result.lockDeviceData();	// ensure device data isn't freed after we leave scope
	return result;
}


/*
	Finds location of min value in grayscale image on device
*/
cv::Point DeviceImage::minLoc() {
	cv::Point min_point(0, 0);
	// ensure gray image exists
	if (gray == 0)
	{
		std::cerr << "ERR: Can't find minimum since no gray image found" << std::endl;
		return min_point;
	}

	// allocate device memory
	uint * min_ptr_device = 0;
	cudaMalloc((void**)&min_ptr_device, 2 * sizeof(uint));

	// call kernel to process on device
	callMinLocKernel(gray, nRows, nCols, min_ptr_device);

	// get results back on host
	uint * min_ptr_host = (uint *)malloc(2 * sizeof(uint));
	cudaMemcpy(&min_ptr_host[0], &min_ptr_device[0], 2 * sizeof(uint), cudaMemcpyDeviceToHost);

	min_point.x = min_ptr_host[0];
	min_point.y = min_ptr_host[1];
	return min_point;
}