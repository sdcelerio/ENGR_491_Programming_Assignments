# Define the executable
add_executable(LED_Data LED_Data.cpp)

# Link the libraries found in the root
target_link_libraries(LED_Data
    PRIVATE
    dv::processing
    ${OpenCV_LIBS}
    Common_Library
)
