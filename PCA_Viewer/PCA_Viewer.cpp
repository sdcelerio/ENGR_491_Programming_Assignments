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
#define CAMERA_PCA_RATE_MS 10      // How often the program will calculate the PCA and display it

void Draw_Vector(cv::Mat& Frame, double Center_X, double Center_Y, double Vector[], double Magnitude, cv::Scalar Color);

int main(void) {
    // Open the first camera that appears
    dv::io::camera::CameraPtr Camera = dv::io::camera::open();
    
    /*
    // Initialize the reader for the file
    std::filesystem::path filePath = "/mnt/c/Users/sdcel/OneDrive/Documents/School_Work/College_Work/4th_Year/2026_Spring/ENGR_491_512/ENGR_491_Programming_Assignments/data/Ruler.aedat4";
    dv::io::MonoCameraRecording Camera(filePath);
    if (!Camera.isEventStreamAvailable()) {
        std::cerr << "Error! Could not find any events in the filepath " << filePath << std::endl;
        return 1;
    }   
    */
    
    // Intialize the Visualizer
    dv::visualization::EventVisualizer visualizer(Camera->getEventResolution().value(), dv::visualization::colors::black,
        dv::visualization::colors::green, dv::visualization::colors::red);
    
    // Intialize the Event Slicer
    dv::EventStreamSlicer Slicer;
    PCA_Tracker Tracker(ROLLING_WINDOW_SIZE);
    std::chrono::microseconds Rate(CAMERA_PCA_RATE_MS * 1000); 
    Slicer.doEveryTimeInterval(Rate, [&](const dv::EventStore &Events) {
        // Send events to the PCA Tracker for it to calculate the PCA vectors
        Tracker.Accept_Event_Batch(Events);

        // Retrieve the PCA values 
        double Means[2];
        double Eigenvalues[2];
        double Vectors[2][2];
        Tracker.Get_Means(Means[0], Means[1]);
        Tracker.Get_Eigenvalues(Eigenvalues[0], Eigenvalues[1]);
        Tracker.Get_Eigenvectors(Vectors);
        
        // Produce frame and output
        cv::Mat frame = visualizer.generateImage(Events);
        cv::Point center(static_cast<int>(Means[0]), static_cast<int>(Means[1]));
        Tracker.Draw_PCA_Vectors(frame, cv::Scalar(255, 0, 255), cv::Scalar(255, 255, 255), 3);
        cv::circle(frame, center, 5, cv::Scalar(255, 0, 0), -1);
        cv::imshow("Real-time PCA Tracking", frame);
        cv::pollKey();
    });

    // Loop to feed event readings into the slicer 
    while(Camera->isRunning()) {
       if (std::optional<dv::EventStore> Events = Camera->getNextEventBatch()) {
           Slicer.accept(*Events);
        }
    }

    return 0;
}