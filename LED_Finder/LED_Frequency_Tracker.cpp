#include <vector>
#include <dv-processing/core/core.hpp>
#include <dv-processing/cluster/mean_shift.hpp>
#include <opencv2/opencv.hpp>
#include "LED_Frequency_Tracker.hpp"

LED_Frequency_Tracker::LED_Frequency_Tracker(const cv::Size    Resolution,
                            double             Target_Frequency,
                            double             Tolerance,
                            int                Required_Matches,
                            const cv::Vec3b   Tracker_Color,
                            int                Minimum_Cluster_Points,
                            float              Cluster_Search_Radius) 
    : Detector          (Resolution, Target_Frequency, Tolerance, Required_Matches)
    , Tracker_Color             (Tracker_Color)
    , Minimum_Cluster_Points (Minimum_Cluster_Points)
    , Cluster_Search_Radius (Cluster_Search_Radius)
{}

void LED_Frequency_Tracker::Accept_Event_Batch(const dv::EventStore& Events, cv::Mat& Frame) {
    // Check if there are any events in the event store. Used to prevent buggy behavior on empty event stores
    if (Events.isEmpty()) 
        return;

    // Pass events into frequency detector and extract event store to feed into the cluster algorithm
    this->Detector.Accept_Event_Batch(Events);
    dv::EventStore temp(this->Detector.Generate_Events());
    if (temp.isEmpty())
        return;
    dv::cluster::mean_shift::MeanShiftEventStoreAdaptor meanShift(temp, this->Cluster_Search_Radius, 0.01f, 500, 30);
    auto [Cluster_Centers, Labels, Counts, Variances] = meanShift.fit();

    // Draw the bounding box and highlight the detected pixels
    this->Detector.Highlight_Pixels(Frame, this->Tracker_Color);
    for (int i = 0; i < Cluster_Centers.size(); i++) {
        if (Counts.at(i) <= this->Minimum_Cluster_Points)
        continue;
        
        const int halfSize = 15;
        cv::Point2f pt(Cluster_Centers[i].pt.x(), Cluster_Centers[i].pt.y());
        cv::Point2f topLeft  = pt - cv::Point2f(halfSize, halfSize);
        cv::Point2f botRight = pt + cv::Point2f(halfSize, halfSize);
        cv::rectangle(Frame,
            pt - cv::Point2f(halfSize, halfSize),
            pt + cv::Point2f(halfSize, halfSize),
            this->Tracker_Color);
        cv::putText(Frame,
            "400 Hz",                                  
            topLeft - cv::Point2f(0, 5),                
            cv::FONT_HERSHEY_SIMPLEX,
            0.4,                                  
            this->Tracker_Color,
            1);                    
    }
}
