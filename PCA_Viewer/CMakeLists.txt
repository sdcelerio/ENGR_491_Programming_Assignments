# Define the executable
add_executable(run PCA_Viewer.cpp)

# Link the libraries found in the root
target_link_libraries(run
    PRIVATE
    dv::processing
    ${OpenCV_LIBS}
    Common_Library
)