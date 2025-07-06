#include "hogdetector.h"
#include <opencv2/ml.hpp>
#include <iostream>

HOGFaceDetector::HOGFaceDetector() {
    // Set HOG parameters
    hog = cv::HOGDescriptor(cv::Size(64, 64),
                            cv::Size(16, 16),
                            cv::Size(8, 8),
                            cv::Size(8, 8),
                            9);
    // Load your trained SVM model
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("face_detector_svm.yml");

    // Convert SVM to HOG Detector vector
    cv::Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    cv::Mat alpha, svidx;
    double rho = svm->getDecisionFunction(0, alpha, svidx);

    std::vector<float> detector(sv.cols + 1);
    memcpy(detector.data(), sv.ptr(), sv.cols * sizeof(float));
    detector[sv.cols] = (float)-rho;

    hog.setSVMDetector(detector);
}

std::vector<BoundingBox> HOGFaceDetector::detectFaces(const cv::Mat& frame) {
    std::vector<cv::Rect> faces;
    hog.detectMultiScale(frame, faces, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2);

    std::vector<BoundingBox> result;
    for (const auto& face : faces) {
        result.push_back({ face.x, face.y, face.width, face.height });
    }
    return result;
}
