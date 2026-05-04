#pragma once

#include <vector>
#include <cstdint>
#include <dv-processing/core/core.hpp>
#include <opencv2/core.hpp>

class Frequency_Estimator {
    /* Private data members */
    private:
        int Width;
        std::vector<std::int64_t> Last_Pos_Timestamp;
        double Measured_Frequency = 0.0;

    /* Public functions */
    public:
        Frequency_Estimator(cv::Size Resolution);

        double Estimate(const dv::EventStore& Events);
        double Get_Frequency() const;
        void Reset();
};