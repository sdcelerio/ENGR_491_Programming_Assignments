cmake_minimum_required(VERSION 3.10)
project(EventPCA)

# Set C++ standard to 17 or 20 for filesystem and dv-processing support
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the required libraries
find_package(dv-processing REQUIRED)
find_package(OpenCV REQUIRED)

# Add subdirectories for each assignment
add_subdirectory(PCA_Viewer)