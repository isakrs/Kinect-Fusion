cmake_minimum_required(VERSION 3.0)
project(exercise_1)
set(CMAKE_CXX_STANDARD 11)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)


find_path(
        FREEIMAGE_INCLUDE_DIR
        NAMES FreeImage.h
        PATHS
        /usr/include
        /usr/local/include
        /sw/include
        /opt/local/include
        DOC "The directory where FreeImage.h resides")

# Find library files
find_library(
        FREEIMAGE_LIBRARY
        NAMES freeimage
        PATHS
        /usr/lib64
        /usr/lib
        /usr/local/lib64
        /usr/local/lib
        /sw/lib
        /opt/local/lib
        ${FREEIMAGE_ROOT_DIR}/lib
        DOC "The FreeImage library")

add_executable(exercise_1 main.cpp FreeImageHelper.cpp)
target_link_libraries (exercise_1 Eigen3::Eigen freeimage)