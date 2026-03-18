#pragma once

#include <vector>
#include <cstdint>
#include <opencv2/core.hpp>

class LED_Tracker {
    /* Public Defined Structures */
    public:
        enum class Status { Tracking, Lost };

        struct TrackedLED {
            cv::Point2d Centroid    = {0.0, 0.0};   // Current centroid position
            cv::Point2d Velocity    = {0.0, 0.0};   // Exponential moving average velocity (pixels/ms)
            cv::Rect    Bounding_Box;                // Axis-aligned search region
            int         Event_Count = 0;             // Valid pixels inside box this cycle
            Status      State       = Status::Lost;  // Current tracking status
        };

    /* Private Data Members */
    private:
        // Configuration
        cv::Size    Resolution;
        int         Target_Cluster_Count;
        int         Box_Padding;                // Extra pixels around a cluster when building a bounding box
        int         Epsilon_Min;                // Binary search lower bound for DBSCAN epsilon
        int         Epsilon_Max;                // Binary search upper bound for DBSCAN epsilon
        int         Min_Cluster_Size;           // Minimum points to consider a DBSCAN cluster valid (filters noise)
        int         Lost_Threshold;             // Event count below which a tracked LED is considered lost
        double      Velocity_Smoothing;         // Exponential moving average alpha (0,1]. Higher = more responsive
        int         Max_Search_Iterations;      // Maximum iterations for the epsilon binary search

        // State
        bool        Initialized = false;
        std::vector<TrackedLED> LEDs;

    /* Public Functions */
    public:
        /**
         * Constructs an LED_Tracker.
         * @param Resolution            Camera resolution (width x height)
         * @param Target_Cluster_Count  Number of LEDs to track
         * @param Box_Padding           Extra pixels added around each cluster bounding box
         * @param Epsilon_Min           Minimum epsilon for DBSCAN binary search
         * @param Epsilon_Max           Maximum epsilon for DBSCAN binary search
         * @param Min_Cluster_Size      Minimum points for a DBSCAN cluster to be considered valid
         * @param Lost_Threshold        Event count below which an LED is marked lost
         * @param Velocity_Smoothing    EMA alpha for velocity updates (0,1]
         */
        LED_Tracker(cv::Size Resolution,
                    int      Target_Cluster_Count,
                    int      Box_Padding           = 20,
                    int      Epsilon_Min            = 3,
                    int      Epsilon_Max            = 80,
                    int      Min_Cluster_Size       = 10,
                    int      Lost_Threshold         = 5,
                    double   Velocity_Smoothing     = 0.3,
                    int      Max_Search_Iterations  = 15);

        /**
         * Main update call. Pass in the valid pixels from the Frequency_Detector each cycle.
         * Internally decides whether to run initialization or tracking mode.
         */
        void Update(const std::vector<cv::Point2i>& Valid_Pixels);

        /**
         * Returns a const reference to the tracked LED states.
         */
        const std::vector<TrackedLED>& Get_LEDs() const;

        /**
         * Returns true if the tracker has completed initialization and is in tracking mode.
         */
        bool Is_Initialized() const;

        /**
         * Forces a re-initialization on the next Update() call.
         */
        void Reset();

        /**
         * Draws bounding boxes, centroids, and velocity vectors onto the given frame.
         */
        void Draw(cv::Mat& Frame, cv::Scalar Box_Color = cv::Scalar(0, 255, 0), cv::Scalar Velocity_Color = cv::Scalar(255, 0, 0)) const;

    /* Private Helper Functions */
    private:
        /**
         * Runs the DBSCAN binary search to find epsilon that yields Target_Cluster_Count clusters.
         * Returns true if initialization succeeded.
         */
        bool Initialize(const std::vector<cv::Point2i>& Valid_Pixels);

        /**
         * Per-box tracking update: gathers pixels in each box, computes centroids,
         * updates velocity, shifts boxes. Falls back to re-init if too many LEDs are lost.
         */
        void Track(const std::vector<cv::Point2i>& Valid_Pixels);

        /**
         * Builds an axis-aligned bounding box around a set of points with padding,
         * clamped to the camera resolution.
         */
        cv::Rect Build_Bounding_Box(const std::vector<cv::Point2i>& Cluster_Points) const;

        /**
         * Computes the centroid of the given points.
         */
        static cv::Point2d Compute_Centroid(const std::vector<cv::Point2i>& Points);
};
