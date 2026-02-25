#pragma once

#include <vector>
#include <cstdint>
#include <dv-processing/core/core.hpp>
#include <opencv2/opencv.hpp>

class Frequency_Detector {
    /* Private defined structs */
    private:
        struct PixelState {
            std::int64_t Latest_Timestamp = 0;
            int Num_Matches = 0;
            bool Is_Valid = false;
        };
    
    /* Private data members */
    private:
        cv::Size Size;
        double Target_Frequency;
        double Tolerance;
        int Required_Matches;

        // 1D arrays flattened from 2D (width * height) for cache performance
        std::vector<PixelState> Pixel_States;
        std::vector<std::uint32_t> Valid_Indexes;
    
    /* Public functions */
    public:
        /**
         * Constructs a Frequency_Detector Object given the camera resolution and parameters for valid oscillating pixels
         */
        Frequency_Detector(cv::Size Size, double Target_Frequency, double Tolerance, int Required_Matches);

        /**
         * Processes an incoming batch of events and updates a binary OpenCV mask.
         */
        void Accept_Event_Batch(const dv::EventStore& Events);

        /**
         * Draws the detected pixels on to the given frame with the given color
         */
        void Highlight_Pixels(cv::Mat& Frame, cv::Vec3b Color) const;

    /* Private helper functions */
    private:
        /**
         * Given the latest timestamp of the event batch, the old validated pixels are removed from the Pixel State Vector
         */
        void Remove_Old_Pixels(std::int64_t Latest_Timestamp);
};