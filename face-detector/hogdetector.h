#ifndef HOGDETECTOR_H
#define HOGDETECTOR_H

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
