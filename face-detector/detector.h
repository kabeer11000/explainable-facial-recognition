#ifndef HOG_DETECTOR_H
#define HOG_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

struct BoundingBox {
int x;
int y;
int width;
int height;
};

class HOGFaceDetector {
public:
HOGFaceDetector();
std::vector<BoundingBox> detectFaces(const cv::Mat& frame);

private:
cv::HOGDescriptor hog;
};

#endif
