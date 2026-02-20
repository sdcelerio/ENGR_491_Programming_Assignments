# Define the library
add_library(Common_Library 
    PCA_Tracker.cpp
    Frequency_Detector.cpp
)

# The library needs OpenCV/DV too
find_package(OpenCV REQUIRED)
find_package(dv-processing REQUIRED)

# Link the libraries needed for the common library files
target_link_libraries(Common_Library 
    PRIVATE 
    ${OpenCV_LIBS} 
    dv::processing
)

# Allows other folders to see .hpp files provided by the common library
target_include_directories(Common_Library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})