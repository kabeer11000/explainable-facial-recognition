#include "hog_detector.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
HOGFaceDetector detector;
cv::Mat frame = cv::imread("sample.jpg");

if (frame.empty()) {
std::cerr << "Failed to load image." << std::endl;
return -1;
}

auto boxes = detector.detectFaces(frame);

for (const auto& box : boxes) {
cv::rectangle(frame, cv::Rect(box.x, box.y, box.width, box.height), cv::Scalar(255, 0, 0), 2);
}

cv::imshow("HOG Face Detection", frame);
cv::waitKey(0);

return 0;
}
