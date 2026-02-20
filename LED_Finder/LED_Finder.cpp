#include <iostream>
#include <chrono>
#include <filesystem>
#include <dv-processing/core/core.hpp>
#include <dv-processing/io/camera/discovery.hpp>            // Used for real-time readings
#include <dv-processing/io/mono_camera_recording.hpp>       // Used for reading .aedat4 recordings
#include <dv-processing/core/stream_slicer.hpp>             // Used to collect readings 
#include <dv-processing/visualization/event_visualizer.hpp> // Used to generate images to display
#include <opencv4/opencv2/highgui.hpp>                      // Used to display the data
#include "Frequency_Detector.hpp"

#define CAMERA_RATE_MS 10      // How often the program will calculate and display it

int main(void) {
    // Initialize the reader for the file
    dv::io::camera::CameraPtr Camera = dv::io::camera::open();
    
    /*
    std::filesystem::path filePath = "../data/LEDs_Fast.aedat4";
    dv::io::MonoCameraRecording Reader(filePath);
    dv::io::MonoCameraRecording* Camera = &Reader;
    if (!Camera->isEventStreamAvailable()) {
        std::cerr << "Error! Could not find any events in the filepath " << filePath << std::endl;
        return 1;
    }   
    */
    // Get the camera resolution
    auto resolution = Camera->getEventResolution();
    if (!resolution.has_value()) {
        std::cerr << "Camera does not provide event resolution!" << std::endl;
        return 1;
    }

    // Initialize detectors
    int16_t width = resolution->width;
    int16_t height = resolution->height;
    Frequency_Detector Detector_100(width, height, 100.0, 10.0, 3);
    Frequency_Detector Detector_200(width, height, 200.0, 20.0, 3);
    Frequency_Detector Detector_300(width, height, 300.0, 30.0, 3);
    Frequency_Detector Detector_400(width, height, 400.0, 40.0, 3);
    
    // Initalize visualizer
    dv::visualization::EventVisualizer visualizer(Camera->getEventResolution().value(), dv::visualization::colors::black,
        dv::visualization::colors::green, dv::visualization::colors::red);
    cv::namedWindow("Events", cv::WINDOW_NORMAL);
    cv::namedWindow("Detected LEDs", cv::WINDOW_NORMAL);
    
    // Intialize the Event Slicer
    cv::Mat detectionMask(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    dv::EventStreamSlicer Slicer;
    std::chrono::microseconds Rate(CAMERA_RATE_MS * 1000); 
    Slicer.doEveryTimeInterval(Rate, [&](const dv::EventStore &Events) {
        
        //std::this_thread::sleep_for(std::chrono::microseconds(CAMERA_RATE_MS * 1000));
    });

    std::cout << "Starting live capture. Press 'q' or 'ESC' to exit." << std::endl;
    while (Camera->isRunning()) {
        if (std::optional<dv::EventStore> Events = Camera->getNextEventBatch())
           // Read a batch of events (e.g., sliced roughly every 30ms by default)
        if (!Events->isEmpty()) {
            // Process the events and generate a mask
            detectionMask.setTo(cv::Scalar(0, 0, 0));
            Detector_100.Accept_Event_Batch(*Events);
            Detector_100.Highlight_Pixels(detectionMask, cv::Vec3b(255, 255, 255));
            
            Detector_200.Accept_Event_Batch(*Events);
            Detector_200.Highlight_Pixels(detectionMask, cv::Vec3b(0, 0, 255));

            Detector_300.Accept_Event_Batch(*Events);
            Detector_300.Highlight_Pixels(detectionMask, cv::Vec3b(0, 255, 0));

            Detector_400.Accept_Event_Batch(*Events);
            Detector_400.Highlight_Pixels(detectionMask, cv::Vec3b(255, 0, 0));

            cv::imshow("Events", visualizer.generateImage(*Events));
            cv::imshow("Detected LEDs", detectionMask);
        }

        cv::waitKey(1);
    }

    return 0;
}