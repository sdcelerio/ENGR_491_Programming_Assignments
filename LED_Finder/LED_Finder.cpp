#include <iostream>
#include <chrono>
#include <filesystem>
#include <dv-processing/core/core.hpp>
#include <dv-processing/io/camera/discovery.hpp>            // Used for real-time readings
#include <dv-processing/io/mono_camera_recording.hpp>       // Used for reading .aedat4 recordings
#include <dv-processing/visualization/event_visualizer.hpp> // Used to generate images to display
#include <opencv2/opencv.hpp>                               // Used to display the data
#include "LED_Frequency_Tracker.hpp"
#include "PCA_Tracker.hpp"
#include "DBSCAN_Grid.hpp"

#define CAMERA_RATE_MS 10      // How often the program will calculate and display it

int main(void) {
    // Initialize the live camera
    dv::io::camera::CameraPtr Camera = dv::io::camera::open();
    
    // Get the camera resolution
    auto resolution = Camera->getEventResolution();
    if (!resolution.has_value()) {
        std::cerr << "Camera does not provide event resolution!" << std::endl;
        return 1;
    }

    // Check if event stream is avaiable
    if (!Camera->isEventStreamAvailable()) {
        std::cerr << "Camera does not provide event stream!" << std::endl;
        return 1;
    }

    // Initialize led frequency tracker
    //LED_Frequency_Tracker LED_300_Tracker(*resolution, 300.0, 30.0, 3, cv::Vec3b(255, 255, 255), 100, 40); 
    PCA_Tracker Axis(3);
    Frequency_Detector Tracker_1000(*resolution, 1000, 100, 3);
    // Initalize visualizer
    dv::visualization::EventVisualizer visualizer(Camera->getEventResolution().value(), dv::visualization::colors::black,
        dv::visualization::colors::green, dv::visualization::colors::red);
    cv::namedWindow("Events", cv::WINDOW_NORMAL);
    cv::namedWindow("Detected LEDs", cv::WINDOW_NORMAL);
    cv::Mat detectionMask(*resolution, CV_8UC3, cv::Vec3b(0, 0, 0));

    std::cout << "Starting live capture." << std::endl;
    while (Camera->isRunning()) {
        if (std::optional<dv::EventStore> Events = Camera->getNextEventBatch()) {
            // Read a batch of events
            if (!Events->isEmpty()) {
                // Process the events and generate a mask
                detectionMask.setTo(cv::Vec3b(0, 0, 0));
                Tracker_1000.Accept_Event_Batch(*Events);
                Tracker_1000.Highlight_Pixels(detectionMask, cv::Vec3b(255, 255, 255));

                DBSCAN_Grid Test(Tracker_1000.Get_Valid_Pixels(), *resolution, 20, 20);
                auto [Cluster_Centers, Labels, Counts] = Test.Fit();

                dv::EventStore temp;
                for (cv::Point2i center : Cluster_Centers) {
                    temp.emplace_back(0, center.x, center.y, true);
                }
                Axis.Accept_Event_Batch(temp);
                Axis.Draw_PCA_Vectors(detectionMask, cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), 1);

                for (int i = 0; i < Cluster_Centers.size(); i++) {
                    const int halfSize = 15;
                    cv::Point2f pt(Cluster_Centers[i].x, Cluster_Centers[i].y);
                    cv::Point2f topLeft  = pt - cv::Point2f(halfSize, halfSize);
                    cv::Point2f botRight = pt + cv::Point2f(halfSize, halfSize);
                    cv::rectangle(detectionMask,
                        pt - cv::Point2f(halfSize, halfSize),
                        pt + cv::Point2f(halfSize, halfSize),
                        cv::Vec3b(255, 255, 255)); 
                }

                cv::imshow("Events", visualizer.generateImage(*Events));
                cv::imshow("Detected LEDs", detectionMask);
            }
            //std::this_thread::sleep_for(std::chrono::microseconds(Events->duration()));
        }
        cv::pollKey();
    }

    return 0;
}