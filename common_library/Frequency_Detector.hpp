#include <stdint.h>
#include <vector>
#include <dv-processing/core/core.hpp>

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

        // 1D arrays flattened from 2D (width * height) for performance
        std::vector<PixelState> Pixel_States;
        std::vector<int64_t> Last_Timestamps;
        std::vector<int> Consecutive_Matches;
    
    /* Public functions */
    public:
        /**
         * @param w Camera width
         * @param h Camera height
         * @param freq Target frequency in Hz
         * @param tol Frequency tolerance in Hz
         * @param matches Consecutive cycles required to confirm a detection
         */
        Frequency_Detector(int16_t Width, int16_t Height, double Target_Frequency, double Tolerance, int Required_Matches);

        /**
         * Processes an incoming batch of events and updates a binary OpenCV mask.
         */
        void Accept_Event_Batch(const dv::EventStore& Events, cv::Mat& outputMask);

        /**
         * Draws the detected pixels on to the given frame
         */
        void Highlight_Pixels(cv::Mat& Frame, cv::Vec3b Color);
};