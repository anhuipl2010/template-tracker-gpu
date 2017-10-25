#include <iostream>
#include <iomanip>
#include <ctime>

#include <opencv2/opencv.hpp>

#include "DeviceImage.h"


// GLOBAL INFO FOR INPUT IMAGES
const int NUM_IMAGES = 252;
std::string IMAGE_DIR;

// hard-coded data from ground truth file
cv::Point2i frame_one_truth[4] = {
	cv::Point2i(6, 193),
	cv::Point2i(6, 166),
	cv::Point2i(49, 166),
	cv::Point2i(49, 193)
};


/*
	Prints duration between two clock events
*/
void printDuration(clock_t start, clock_t stop) {
	double total_time = (stop - start) / (double)CLOCKS_PER_SEC;
	std::cout << "Processing time : " << total_time * 1000.0 << "ms" << std::endl;
}

int main(int argc, char **argv)
{
	// parse input params
	if (argc != 2) {
		std::cout << "USAGE: uber_project.exe IMAGE_DIRECTORY" << std::endl;
		return -1;
	}
	else {
		IMAGE_DIR = std::string(argv[1]);
		if (IMAGE_DIR.back() != '\\')
			IMAGE_DIR.append("\\");
	}

	// warmup the GPU (takes about one second)
	std::cout << "Warming up GPU... ";
	gpuWarmup(1);
	std::cout << "DONE!" << std::endl;

	// initialize car bounding box
	cv::Point bbox[2] = {
		frame_one_truth[1],	// top left
		frame_one_truth[3]	// bottom right
	};

	// initialize region of interest vars beyond scope of for-loop
	uint roi_width, roi_height;
	DeviceImage lastFrameROI;

	clock_t timer_start, timer_stop;
	std::cout << "Increment tracker by pressing any key.... " << std::endl;
	for (int im_num = 1; im_num <= NUM_IMAGES; im_num++) {
		
		// get image path
		std::stringstream ss;
		ss << std::setw(8) << std::setfill('0') << im_num;
		std::string img_path = IMAGE_DIR + ss.str() + ".jpg";

		// read image
		cv::Mat img = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
		if (!img.data) {
			std::cerr << "Failed to open '" << img_path << "'" << std::endl;
			return -1;
		}
		
		// start timer
		timer_start = clock(); 

		// get gray image on device
		DeviceImage d;
		d.upload(img);
		d.convertToGrayscale();

		// do template matching with prev frame's ROI
		cv::Point min_match(0, 0);
		if (im_num > 1)
		{
			// do template matching
			DeviceImage match = d.templateMatch(lastFrameROI);
			min_match = match.minLoc();		// THIS IS REALLY SLOW -> SHOULD IMPLEMENT REDUCTION
			match.unlockDeviceData();		// tell system that dev data can be freed

			// update bounding box
			bbox[0] = min_match;
			bbox[1] = cv::Point(min_match.x + roi_width, min_match.y + roi_height);
		}

		// get car image ROI
		roi_width = bbox[1].x - bbox[0].x;
		roi_height = bbox[1].y - bbox[0].y;
		lastFrameROI = d.getROI(bbox[0], roi_width, roi_height);

		// stop timer
		timer_stop = clock();
		printDuration(timer_start, timer_stop);

		// display tracking results
		cv::rectangle(img, cv::Rect(bbox[0].x, bbox[0].y, roi_width, roi_height), cv::Scalar(0, 0, 255), 1);
		cv::imshow("Tracked Frame", img);
		cv::waitKey(0);
	}

	// allow dev memory to be freed
	lastFrameROI.unlockDeviceData();

	return 0;
}