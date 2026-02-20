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

class LedFrequencyDetector {
private:
    int16_t width;
    int16_t height;
    double targetFreq;
    double tolerance;
    int requiredMatches;

    // 1D arrays flattened from 2D (width * height) for performance
    std::vector<int64_t> lastTimestamps;
    std::vector<int> consecutiveMatches;

public:
    /**
     * @param w Camera width
     * @param h Camera height
     * @param freq Target frequency in Hz
     * @param tol Frequency tolerance in Hz
     * @param matches Consecutive cycles required to confirm a detection
     */
    LedFrequencyDetector(int16_t w, int16_t h, double freq, double tol = 5.0f, int matches = 3)
        : width(w), height(h), targetFreq(freq), tolerance(tol), requiredMatches(matches) {
        
        // Initialize state arrays
        lastTimestamps.resize(width * height, 0);
        consecutiveMatches.resize(width * height, 0);
    }

    /**
     * Processes an incoming batch of events and updates a binary OpenCV mask.
     */
    void processEvents(const dv::EventStore& Events, cv::Mat& outputMask) {
        // Reset the mask for the current batch
        outputMask = cv::Mat::zeros(height, width, CV_8UC1);
        
        for (const auto& Event : Events) {
            // We only measure positive polarity (OFF to ON transitions) to capture full cycles
            if (!Event.polarity())
                continue;

            int index = Event.y() * width + Event.x();
            int64_t dt = Event.timestamp() - lastTimestamps[index];
            lastTimestamps[index] = Event.timestamp();

            // Ignore impossibly short deltas to filter out hardware noise bursts
            if (dt < 1000) 
                continue;

            // After the dt < 1000 guard:
            float measuredFreq = 1e6f / static_cast<float>(dt); // dt is in microseconds

            if (std::abs(measuredFreq - targetFreq) <= tolerance) {
                consecutiveMatches[index]++;
            } else {
                consecutiveMatches[index] = 0; // reset streak on mismatch
            }
            
            
        }

        for (int i = 0; i < width * height; i++) {
            if (consecutiveMatches[i] >= requiredMatches) {
                int x = i % width, y = i / width;
                outputMask.at<uint8_t>(y, x) = 255;
            }
        }
    }
};

int main(void) {
    
    // Initialize the reader for the file
    /*
    dv::io::camera::CameraPtr Camera = dv::io::camera::open();
    */
    
    std::filesystem::path filePath = "../data/LEDs_Slow.aedat4";
    dv::io::MonoCameraRecording Reader(filePath);
    dv::io::MonoCameraRecording* Camera = &Reader;
    if (!Camera->isEventStreamAvailable()) {
        //std::cerr << "Error! Could not find any events in the filepath " << filePath << std::endl;
        return 1;
    }   
    
    auto resolution = Camera->getEventResolution();
    if (!resolution.has_value()) {
        std::cerr << "Camera does not provide event resolution!" << std::endl;
        return 1;
    }

    // Initialize detector
    int16_t width = resolution->width;
    int16_t height = resolution->height;
    LedFrequencyDetector detector(width, height, 4.0, 0.5, 3);
    
    cv::Mat detectionMask;
    dv::visualization::EventVisualizer visualizer(Camera->getEventResolution().value(), dv::visualization::colors::black,
        dv::visualization::colors::green, dv::visualization::colors::red);
    cv::namedWindow("Events", cv::WINDOW_NORMAL);
    cv::namedWindow("Detected LEDs", cv::WINDOW_NORMAL);


    // Intialize the Event Slicer
    dv::EventStreamSlicer Slicer;
    std::chrono::microseconds Rate(100 * 1000); 
    Slicer.doEveryTimeInterval(Rate, [&](const dv::EventStore &Events) {
        // Read a batch of events (e.g., sliced roughly every 30ms by default)
        if (!Events.isEmpty()) {
            // Process the events and generate a mask
            detector.processEvents(Events, detectionMask);
            cv::imshow("Events", visualizer.generateImage(Events));
            cv::imshow("Detected LEDs", detectionMask);
        }

        cv::pollKey();
    });



    std::cout << "Starting live capture. Press 'q' or 'ESC' to exit." << std::endl;
    while (Camera->isRunning()) {
        if (std::optional<dv::EventStore> Events = Camera->getNextEventBatch())
           Slicer.accept(*Events);
        
        std::this_thread::sleep_for(std::chrono::microseconds(10000));
    }

    return 0;
}