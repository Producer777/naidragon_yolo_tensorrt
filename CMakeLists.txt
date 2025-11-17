cmake_minimum_required(VERSION 3.18)
project(naidragon_yolov8_trt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)

# TensorRT
set(TENSORRT_INCLUDE_DIR /usr/include/x86_64-linux-gnu)
set(TENSORRT_LIB_DIR     /usr/lib/x86_64-linux-gnu)

include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIB_DIR})

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

add_executable(yolov8_trt src/main.cpp)

target_link_libraries(yolov8_trt
    ${OpenCV_LIBS}
    nvinfer nvinfer_plugin
    ${CUDA_LIBRARIES}
)

add_executable(precision_cmpr src/precision.cpp)

target_link_libraries(precision_cmpr
    ${OpenCV_LIBS}
    nvinfer nvinfer_plugin
    ${CUDA_LIBRARIES}
)