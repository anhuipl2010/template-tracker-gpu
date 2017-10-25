#ifndef _DEVICE_IMAGE_H_
#define _DEVICE_IMAGE_H_

#include <cstdio>
#include <opencv2/opencv.hpp>

#include "gpu_func.h"


/*
	Class for keeping track of images on GPU device
	and doing necessary processing on device data
*/
class DeviceImage
{
public:
	enum ImageType
	{
		GRAY,
		COLOR
	};

	DeviceImage()
	{
		nRows = 0;	// init member vars
		nCols = 0;
		bgr = 0;
		gray = 0;
		keepDevData = false;
	};
	DeviceImage(int _nRows, int _nCols, uchar3 *_bgr, float *_gray)
	{
		nRows = _nRows;	// init member vars
		nCols = _nCols;
		bgr = _bgr;
		gray = _gray;
		keepDevData = false;
	};
	~DeviceImage()
	{
		if (!keepDevData) {
			if (bgr != 0)  { cudaFree(bgr); }	// free device memory on destruction
			if (gray != 0) { cudaFree(gray); }
		}
	};

	void upload(cv::Mat& host_img);
	cv::Mat download(ImageType imType);

	void convertToGrayscale();
	DeviceImage getROI(cv::Point top_left, int width, int height);
	DeviceImage templateMatch(DeviceImage& tm);
	cv::Point minLoc();

	// control if dev data gets freed when host var destructs
	void lockDeviceData() { 
		if (!keepDevData) { keepDevData = true; }
	}
	void unlockDeviceData() {
		if (keepDevData) { keepDevData = false; }
	}

	float *grayPtr() { return gray; };
	uchar3 *bgrPtr() { return bgr; };
	uint rows() { return nRows; };
	uint cols() { return nCols; };

private:
	uint nRows;			// size of image
	uint nCols;
	uchar3 *bgr;		// ptr to BGR-structured image (on device)
	float *gray;		// ptr to grayscale image (on device)
	bool keepDevData;	// decides if device data should freed upon destruction 
};


#endif