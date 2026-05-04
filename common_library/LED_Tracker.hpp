#pragma once
#include <vector>
#include <cstdint>
#include <dv-processing/core/core.hpp>
#include <opencv2/core.hpp>

class LED_Tracker {
    /* Public Defined Structures */
    public:
        enum class Status { Tracking, Lost };

        struct TrackedLED {
            cv::Point2d Centroid    = {0.0, 0.0};
            cv::Point2d Velocity    = {0.0, 0.0};
            cv::Rect    Bounding_Box;
            int         Event_Count = 0;
            int         Lost_Frames = 0;
            Status      State       = Status::Lost;
        };

    /* Private Data Members */
    private:
        cv::Size    Resolution;
        int         Target_Cluster_Count;
        int         Box_Half_Size;
        int         Min_Hot_Pixels;
        int         Lost_Threshold;
        int         Lost_Grace_Frames;
        double      Velocity_Smoothing;
        double      Expected_Ratio;             // Expected eigenvalue ratio of LED geometry (0 = skip check)
        double      Ratio_Tolerance;            // Allowed deviation from expected ratio

        bool        Initialized = false;
        std::vector<TrackedLED> LEDs;

    /* Public Functions */
    public:
        LED_Tracker(cv::Size Resolution,
                    int      Target_Cluster_Count,
                    int      Box_Half_Size       = 25,
                    int      Min_Hot_Pixels      = 30,
                    int      Lost_Threshold      = 3,
                    int      Lost_Grace_Frames   = 10,
                    double   Velocity_Smoothing  = 0.2,
                    double   Expected_Ratio      = 0.0,
                    double   Ratio_Tolerance     = 0.3);

        void Update(const std::vector<cv::Point2i>& Hot_Pixels);

        /**
         * Filters raw events to only those inside a specific LED's bounding box.
         * Use this to feed per-LED frequency analysis downstream.
         */
        dv::EventStore Get_Events_In_Box(int LED_Index, const dv::EventStore& Events) const;

        const std::vector<TrackedLED>& Get_LEDs() const;
        bool Is_Initialized() const;
        void Reset();
        void Draw(cv::Mat& Frame, cv::Scalar Box_Color = cv::Scalar(0, 255, 0), cv::Scalar Velocity_Color = cv::Scalar(255, 0, 0)) const;

    /* Private Helper Functions */
    private:
        bool Initialize(const std::vector<cv::Point2i>& Hot_Pixels);
        void Track(const std::vector<cv::Point2i>& Hot_Pixels);
        cv::Rect Build_Bounding_Box(cv::Point2d Center) const;
        static cv::Point2d Compute_Centroid(const std::vector<cv::Point2i>& Points);

        /**
         * Computes eigenvalue ratio of cluster centers and compares to Expected_Ratio.
         * Returns true if the geometry matches (or if check is disabled with Expected_Ratio = 0).
         * Requires 3+ cluster centers — returns true for 2 centers (not enough data for ratio).
         */
        bool Validate_Geometry(const cv::Mat& Centers) const;
};
