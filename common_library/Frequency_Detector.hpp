#pragma once

#include <vector>
#include <cstdint>
#include <dv-processing/core/core.hpp>
#include <opencv2/opencv.hpp>

class Frequency_Detector {
    /* Private defined structs */
    private:
        struct PixelState {
            int64_t Latest_Timestamp = 0;
            int Num_Matches = 0;
        };
    
    /* Private data members */
    private:
        int16_t Width;
        int16_t Height;
        double Target_Frequency;
        double Tolerance;
        int Required_Matches;

        // 1D arrays flattened from 2D (width * height) for cache performance
        std::vector<PixelState> Pixel_States;
    
    /* Public functions */
    public:
        Frequency_Detector(int16_t Width, int16_t Height, double Target_Frequency, double Tolerance, int Required_Matches);

        /**
         * Processes an incoming batch of events and updates a binary OpenCV mask.
         */
        void Accept_Event_Batch(const dv::EventStore& Events);

        /**
         * Draws the detected pixels on to the given frame with the given color
         */
        void Highlight_Pixels(cv::Mat& Frame, cv::Vec3b Color);
};