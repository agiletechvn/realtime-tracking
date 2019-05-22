cmake_minimum_required(VERSION 3.2)
project(Tracking)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_COMPILER /usr/local/bin/zapcc++)
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -I/usr/local/opt/llvm/include -I/usr/local/opt/llvm/include/c++/v1/")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
endif()

file(COPY haarcascade_frontalface_default.xml DESTINATION ${PROJECT_BINARY_DIR})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp VideoFaceDetector.cpp VideoFaceDetector.h)
add_executable(demo ${SOURCE_FILES})

include_directories(include)
if(APPLE)
    link_directories(lib/MacOS)
else()
    link_directories(lib/Unix)
endif()

target_link_libraries(demo ${OpenCV_LIBS}
    ssl crypto
    dlib gif png jpeg
    faiss cblas lapack omp)

