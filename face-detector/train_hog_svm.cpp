#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <filesystem>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;
namespace fs = std::filesystem;

// Function to get image file paths from a folder
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
    vector<string> positiveFiles;

    // Load positive images from person1 to person11
    for (int p = 1; p <= 11; ++p) {
        string folder = "dataset/person" + to_string(p) + "/";
        auto files = getImagesFromFolder(folder);
        cout << "Found " << files.size() << " images in " << folder << endl;
        positiveFiles.insert(positiveFiles.end(), files.begin(), files.end());
    }

    // Load negative images
    auto negativeFiles = getImagesFromFolder("dataset/negatives/");
    cout << "Found " << negativeFiles.size() << " negative images." << endl;

    // Define HOG Descriptor parameters
    HOGDescriptor hog(Size(64, 64), Size(16, 16), Size(8, 8), Size(8, 8), 9);

    Mat trainingData;
    vector<int> labels;

    // Process positive images
    for (const auto& file : positiveFiles) {
        Mat img = imread(file, IMREAD_GRAYSCALE);
        if (img.empty()) continue;
        resize(img, img, Size(64, 64));
        vector<float> descriptors;
        hog.compute(img, descriptors);
        trainingData.push_back(Mat(descriptors).reshape(1, 1));
        labels.push_back(+1);
    }

    // Process negative images
    for (const auto& file : negativeFiles) {
        Mat img = imread(file, IMREAD_GRAYSCALE);
        if (img.empty()) continue;
        resize(img, img, Size(64, 64));
        vector<float> descriptors;
        hog.compute(img, descriptors);
        trainingData.push_back(Mat(descriptors).reshape(1, 1));
        labels.push_back(-1);
    }

    // Convert labels vector to a Mat
    Mat labelsMat(labels, true);

    // Setup and train the SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));

    cout << "Training SVM on " << trainingData.rows << " samples..." << endl;
    svm->train(trainingData, ROW_SAMPLE, labelsMat);

    // Save trained model
    svm->save("face_detector_svm.yml");
    cout << "✅ Training complete! Model saved as face_detector_svm.yml" << endl;

    return 0;
}
