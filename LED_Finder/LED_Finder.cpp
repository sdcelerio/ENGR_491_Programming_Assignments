#include <iostream>
#include <chrono>
#include <filesystem>
#include <dv-processing/core/core.hpp>
#include <dv-processing/io/camera/discovery.hpp>            // Used for real-time readings
#include <dv-processing/io/mono_camera_recording.hpp>       // Used for reading .aedat4 recordings
#include <dv-processing/core/stream_slicer.hpp>             // Used to collect readings 
#include <dv-processing/visualization/event_visualizer.hpp> // Used to generate images to display
#include <opencv4/opencv2/highgui.hpp>                      // Used to display the data
#include "PCA_Tracker.hpp"

#define ROLLING_WINDOW_SIZE 20000   // How many recent events the PCA calculation will remember
#define CAMERA_PCA_RATE_MS 10      // How often the program will calculate the PCA and display i

int main(void) {
    
    // Initialize the reader for the file
    std::filesystem::path filePath = "../data/LED_4_Fast.aedat4";
    dv::io::MonoCameraRecording Reader(filePath);
    dv::io::MonoCameraRecording* Camera = &Reader;
    if (!Camera->isEventStreamAvailable()) {
        std::cerr << "Error! Could not find any events in the filepath " << filePath << std::endl;
        return 1;
    }   
    
    // Intialize the Visualizer
    dv::visualization::EventVisualizer visualizer(Camera->getEventResolution().value(), dv::visualization::colors::black,
        dv::visualization::colors::green, dv::visualization::colors::red);

    // Loop to feed event readings into the slicer 
    bool First_Batch = true;
    std::chrono::microseconds Prev_Timestamp = std::chrono::microseconds::max();
    while(Camera->isRunning()) {
        std::optional<dv::EventStore> Events = Camera->getNextEventBatch();
        if (!Events.has_value()) 
            continue;
        
        std::cout << (*Events).at(0).timestamp() << std::endl;
        cv::Mat frame = visualizer.generateImage(*Events);
        cv::imshow("Real-time PCA Tracking", frame);
        cv::pollKey();
        std::this_thread::sleep_for(std::chrono::microseconds(10000 * 10));
    }

    return 0;
}