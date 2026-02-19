cmake_minimum_required(VERSION 3.10)
project(EventPCA)

# Set C++ standard 20 for dv-processing support
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the required libraries
find_package(dv-processing REQUIRED)
find_package(OpenCV REQUIRED)

# Create a 'data' folder in the build directory and copy contents from data/
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data 
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)

# Add subdirectories for the library and each assignment
add_subdirectory(common_library)
add_subdirectory(LED_Finder)
add_subdirectory(PCA_Viewer)