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
    float targetFreq;
    float tolerance;
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
    LedFrequencyDetector(int16_t w, int16_t h, float freq, float tol = 5.0f, int matches = 3)
        : width(w), height(h), targetFreq(freq), tolerance(tol), requiredMatches(matches) {
        
        // Initialize state arrays
        lastTimestamps.resize(width * height, 0);
        consecutiveMatches.resize(width * height, 0);
    }

    /**
     * Processes an incoming batch of events and updates a binary OpenCV mask.
     */
    void processEvents(const dv::EventStore& events, cv::Mat& outputMask) {
        // Reset the mask for the current batch
        outputMask = cv::Mat::zeros(height, width, CV_8UC1);

        for (const auto& event : events) {
            // We only measure positive polarity (OFF to ON transitions) to capture full cycles
            if (event.polarity()) {
                int index = event.y() * width + event.x();
                int64_t dt = event.timestamp() - lastTimestamps[index];
                lastTimestamps[index] = event.timestamp();

                // Ignore impossibly short deltas to filter out hardware noise bursts
                if (dt > 1000) { 
                    // Convert delta time (microseconds) to frequency (Hz)
                    float freq = 1000000.0f / static_cast<float>(dt);

                    // Check if the measured frequency falls within our target range
                    if (std::abs(freq - targetFreq) <= tolerance) {
                        consecutiveMatches[index]++;
                    } else {
                        // Reset the match counter if the rhythm breaks
                        consecutiveMatches[index] = 0;
                    }

                    // If a pixel has blinked at the right frequency enough times, mark it
                    if (consecutiveMatches[index] >= requiredMatches) {
                        outputMask.at<uint8_t>(event.y(), event.x()) = 255;
                        
                        // Prevent integer overflow and keep the detection "hot"
                        consecutiveMatches[index] = requiredMatches; 
                    }
                }
            }
        }
    }
};

int main(void) {
    
    // Initialize the reader for the file
    /*
    dv::io::camera::CameraPtr Camera = dv::io::camera::open();
    */


    std::filesystem::path filePath = "../data/LED_4_Slow_Clean.aedat4";
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

    int16_t width = resolution->width;
    int16_t height = resolution->height;

    // Initialize detector for a 100 Hz LED (+/- 5 Hz) requiring 4 consecutive matches
    LedFrequencyDetector detector(width, height, 100.0f, 5.0f, 4);
    cv::Mat detectionMask;
    cv::namedWindow("Detected LEDs", cv::WINDOW_NORMAL);

    std::cout << "Starting live capture. Press 'q' or 'ESC' to exit." << std::endl;
    while (Camera->isRunning()) {
        // Read a batch of events (e.g., sliced roughly every 30ms by default)
        auto Events = Camera->getNextEventBatch();

        if (Events.has_value() && !Events->isEmpty()) {
            // Process the events and generate a mask
            detector.processEvents(Events.value(), detectionMask);
            cv::imshow("Detected LEDs", detectionMask);
        }

        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) { // 27 is ESC
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10000));
    }

    return 0;
}