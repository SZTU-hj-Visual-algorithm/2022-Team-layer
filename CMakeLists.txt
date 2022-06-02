cmake_minimum_required(VERSION 3.19)
project(yolo_opencv)

set(CMAKE_CXX_STANDARD 11)
set(EIGEN3_INCLUDE_DIRS E:/Eigen3.4.0/include/eigen3)
find_package(OpenCV 4.5.4 REQUIRED)
find_package(Eigen3 REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/src DIR_SRCS)
link_directories(${PROJECT_SOURCE_DIR}/src)

file(GLOB_RECURSE sourcefiles "src/*.cpp")
add_executable(yolo_opencv main.cpp)


target_link_libraries(yolo_opencv ${OpenCV_LIBS})
target_link_libraries(yolo_opencv /lib/libMVSDK.so)
target_link_libraries(yolo_opencv yolo_dir)

add_library(yolo_dir ${DIR_SRCS})
