#include "hog_detector.h"

HOGFaceDetector::HOGFaceDetector() {
    // Load pre-trained SVM for face detection
   
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    // Alternatively, you can set your own SVM detector if you have trained one
    // For example, if you have trained a custom SVM for face detection:
    // cv::HOGDescriptor hog;

    std::vector<BoundingBox> HOGFaceDetector::detectFaces(const cv::Mat& frame) {
        std::vector<cv::Rect> faces;

        // Detect faces
        hog.detectMultiScale(frame, faces, 0, cv::Size(30, 30), cv::Size(0, 0), 1.05, 2);

        std::vector<BoundingBox> result;
        for (const auto& face : faces) {
            result.push_back({ face.x, face.y, face.width, face.height });
        }

        return result;
    }
