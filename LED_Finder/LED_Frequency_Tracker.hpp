#pragma once

#include <dv-processing/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "Frequency_Detector.hpp"

class LED_Frequency_Tracker {
    /* Private data members */
    private:
        Frequency_Detector Detector;

        cv::Vec3b  Tracker_Color;
        int         Minimum_Cluster_Points;
        float       Cluster_Search_Radius;
        
    /* Public functions */
    public:
        LED_Frequency_Tracker(const cv::Size    Resolution,
                            double             Target_Frequency,
                            double             Tolerance,
                            int                Required_Matches,
                            const cv::Vec3b    Tracker_Color,
                            int                Minimum_Cluster_Points,
                            float              Cluster_Search_Radius);

        
        void Accept_Event_Batch(const dv::EventStore& Events, cv::Mat& Frame);
};