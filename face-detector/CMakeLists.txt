cmake_minimum_required(VERSION 3.10)
project(FaceDetector)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(TrainHOGSVM train_hog_svm.cpp)
target_link_libraries(TrainHOGSVM ${OpenCV_LIBS})

add_executable(HOGFaceDetector main.cpp hog_detector.cpp)
target_link_libraries(HOGFaceDetector ${OpenCV_LIBS})
add_executable(HOGFaceDetectorGUI main_gui.cpp hog_detector.cpp)
target_link_libraries(HOGFaceDetectorGUI ${OpenCV_LIBS})

