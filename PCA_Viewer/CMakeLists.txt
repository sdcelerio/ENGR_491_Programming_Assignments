# Define the executable
add_executable(PCA_Viewer PCA_Viewer.cpp)

# Link the libraries found in the root
target_link_libraries(PCA_Viewer
    PRIVATE
    dv::processing
    ${OpenCV_LIBS}
)