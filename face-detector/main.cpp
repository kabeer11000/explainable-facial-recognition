#include "hogdetector.h"
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
    HOGFaceDetector detector;

    // Load test image
    Mat frame = imread("sample.jpg");
    if (frame.empty()) {
        cerr << "❌ sample.jpg not found." << endl;
        return -1;
    }

    // Detect faces
    auto boxes = detector.detectFaces(frame);

    // Draw bounding boxes
    for (const auto& box : boxes) {
        rectangle(frame, Rect(box.x, box.y, box.width, box.height), Scalar(0, 255, 0), 2);
    }

    // Display result
    imshow("HOG Face Detection", frame);
    waitKey(0);
    return 0;
}

