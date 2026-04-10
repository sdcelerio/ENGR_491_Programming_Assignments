#pragma once

#include <vector>
#include <cstdint>
<<<<<<< HEAD
=======
#include <dv-processing/core/core.hpp>
>>>>>>> 5616368c25275ce01290fcea60dea72030f5f292
#include <opencv2/core.hpp>

class LED_Tracker {
    /* Public Defined Structures */
    public:
        enum class Status { Tracking, Lost };

        struct TrackedLED {
<<<<<<< HEAD
            cv::Point2d Centroid    = {0.0, 0.0};   // Current centroid position
            cv::Point2d Velocity    = {0.0, 0.0};   // Exponential moving average velocity (pixels/ms)
            cv::Rect    Bounding_Box;                // Axis-aligned search region
            int         Event_Count = 0;             // Valid pixels inside box this cycle
            Status      State       = Status::Lost;  // Current tracking status
=======
            cv::Point2d Centroid    = {0.0, 0.0};
            cv::Point2d Velocity    = {0.0, 0.0};
            cv::Rect    Bounding_Box;
            int         Event_Count = 0;
            int         Lost_Frames = 0;
            Status      State       = Status::Lost;
>>>>>>> 5616368c25275ce01290fcea60dea72030f5f292
        };

    /* Private Data Members */
    private:
<<<<<<< HEAD
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
=======
        cv::Size    Resolution;
        int         Target_Cluster_Count;
        int         Box_Half_Size;
        int         Min_Hot_Pixels;
        int         Lost_Threshold;
        int         Lost_Grace_Frames;
        double      Velocity_Smoothing;
        double      Expected_Ratio;             // Expected eigenvalue ratio of LED geometry (0 = skip check)
        double      Ratio_Tolerance;            // Allowed deviation from expected ratio

>>>>>>> 5616368c25275ce01290fcea60dea72030f5f292
        bool        Initialized = false;
        std::vector<TrackedLED> LEDs;

    /* Public Functions */
    public:
<<<<<<< HEAD
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
=======
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
>>>>>>> 5616368c25275ce01290fcea60dea72030f5f292
        void Draw(cv::Mat& Frame, cv::Scalar Box_Color = cv::Scalar(0, 255, 0), cv::Scalar Velocity_Color = cv::Scalar(255, 0, 0)) const;

    /* Private Helper Functions */
    private:
<<<<<<< HEAD
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
=======
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
>>>>>>> 5616368c25275ce01290fcea60dea72030f5f292
};
