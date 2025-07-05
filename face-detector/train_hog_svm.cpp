#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <filesystem>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;
namespace fs = std::filesystem;

vector<string> getImagesFromFolder(const string& folderPath) {
vector<string> files;
for (const auto& entry : fs::directory_iterator(folderPath)) {
if (entry.is_regular_file()) {
files.push_back(entry.path().string());
}
}
return files;
}

int main() {
vector<string> positiveFiles = getImagesFromFolder("dataset/positives/");
vector<string> negativeFiles = getImagesFromFolder("dataset/negatives/");

vector<Mat> images;
vector<int> labels;

for (const auto& file : positiveFiles) {
Mat img = imread(file, IMREAD_GRAYSCALE);
resize(img, img, Size(64, 64));
images.push_back(img);
labels.push_back(1);
}

for (const auto& file : negativeFiles) {
Mat img = imread(file, IMREAD_GRAYSCALE);
resize(img, img, Size(64, 64));
images.push_back(img);
labels.push_back(0);
}

HOGDescriptor hog(Size(64,64), Size(16,16), Size(8,8), Size(8,8), 9);
Mat trainingData;

for (const auto& img : images) {
vector<float> descriptors;
hog.compute(img, descriptors);
trainingData.push_back(Mat(descriptors).reshape(1, 1));
}

Mat labelsMat(labels, true);

Ptr<SVM> svm = SVM::create();
svm->setType(SVM::C_SVC);
svm->setKernel(SVM::LINEAR);
svm->train(trainingData, ROW_SAMPLE, labelsMat);
svm->save("face_detector_svm.yml");

cout << "Training completed and model saved!" << endl;
return 0;
}
