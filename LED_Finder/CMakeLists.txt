# Define the executable
add_executable(LED_Finder LED_Finder.cpp)

# Link the libraries found in the root
target_link_libraries(LED_Finder
    PRIVATE
    dv::processing
    ${OpenCV_LIBS}
    Common_Library
)