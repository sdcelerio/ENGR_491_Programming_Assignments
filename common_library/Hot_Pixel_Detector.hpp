#pragma once

#include <vector>
#include <cstdint>
#include <dv-processing/core/core.hpp>
#include <opencv2/core.hpp>

class Hot_Pixel_Detector {
    /* Private Data Members */
    private:
        cv::Size Resolution;
        int Threshold;                              // Minimum events per pixel to be considered "hot"
        std::vector<std::int32_t> Counts;           // 1D event count grid (width * height)
        std::vector<cv::Point2i> Hot_Pixels;        // Output: pixels exceeding threshold

    /* Public Functions */
    public:
        /**
         * Constructs a Hot_Pixel_Detector.
         * @param Resolution    Camera resolution
         * @param Threshold     Minimum event count per pixel in one batch to be considered hot
         */
        Hot_Pixel_Detector(cv::Size Resolution, int Threshold = 3);

        /**
         * Processes a batch of raw events. Counts events per pixel and builds the hot pixel list.
         * Call once per batch — this resets counts each time.
         */
        void Process_Batch(const dv::EventStore& Events);

        /**
         * Returns a const reference to the hot pixels found in the last batch.
         */
        const std::vector<cv::Point2i>& Get_Hot_Pixels() const;

        /**
         * Draws the hot pixels onto the given frame.
         */
        void Highlight_Pixels(cv::Mat& Frame, cv::Vec3b Color) const;
};
