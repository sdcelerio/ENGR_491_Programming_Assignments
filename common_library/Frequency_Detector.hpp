#pragma once

#include <vector>
#include <cstdint>
#include <dv-processing/core/core.hpp>
#include <opencv2/core.hpp>

class Frequency_Detector {
    /* Private defined structs */
    private:
        struct PixelState {
            std::int64_t Latest_Timestamp = 0;
            int Num_Matches = 0;
            std::int32_t Index_In_Valid  = -1; // -1 indicates a non-valid node
        };
    
    /* Private data members */
    private:
        cv::Size Resolution;
        double Target_Frequency;
        double Tolerance;
        int Required_Matches;
        std::int64_t Expiry_Threshold;

        // Vector arrays for storing the state of the events
        std::vector<PixelState> Pixel_States;   // 1D array flattened from 2D (width * height) for cache performance
        std::vector<cv::Point2i> Valid_Pixels; // Vector that stores the points of each valid pixel in cv::Point2i format
    
    /* Public functions */
    public:
        /**
         * Constructs a Frequency_Detector Object given the camera resolution and parameters for valid oscillating pixels
         */
        Frequency_Detector(const cv::Size Resolution, double Target_Frequency, double Tolerance, int Required_Matches);

        /**
         * Processes an incoming batch of events and updates a binary OpenCV mask.
         */
        void Accept_Event_Batch(const dv::EventStore& Events);

        /**
         * Stores the Valid Pixels into an dv::EventStore format. Used to work with other dv::processing libraries.
         */
        dv::EventStore Generate_Events();

        /**
         * Returns a constant reference to the valid pixels vector. Note to make sure the Frequency Detector remains in scope when using the reference.
         */
        const std::vector<cv::Point2i>& Get_Valid_Pixels();

        /**
         * Draws the detected pixels on to the given frame with the given color
         */
        void Highlight_Pixels(cv::Mat& Frame, cv::Vec3b Color) const;

    /* Private helper functions */
    private:
        /**
         * Given the latest timestamp of the event batch, the old validated pixels are removed from the Pixel State Vector
         */
        void Remove_Old_Pixels(std::int64_t Latest_Events_Timestamp);

        /**
         * Removes a pixel from Valid_Indexes at the given position using pop-and-swap for O(1) deletion.
         * Updates the swapped pixel's Valid_Index to reflect its new position.
         */
        void Swap_Pop(std::int32_t Target_Index);
};