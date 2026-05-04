#pragma once

#include <vector>
#include <cstdint>
#include <dv-processing/core/core.hpp>
#include <opencv2/core.hpp>

class Kalman_LED_Tracker {
    public:
        enum class Status { Tracking, Lost };

        struct TrackedLED {
            cv::Mat State;                          // 6x1: [px, py, vx, vy, ax, ay]
            cv::Mat Covariance;                     // 6x6
            cv::Point2d Local_PCA_Coords;
            int         Event_Count = 0;
            int         Lost_Frames = 0;
            Status      State_Flag  = Status::Lost;
        };

    private:
        cv::Size    Resolution;
        int         Target_Cluster_Count;
        double      Process_Noise_Sigma;
        double      Measurement_Noise_Sigma;
        double      Gate_Sigma;
        int         Min_Gate_Half_Size;
        int         Max_Gate_Half_Size;
        int         Min_Hot_Pixels;
        int         Lost_Threshold;
        int         Lost_Grace_Frames;
        double      Expected_Ratio;
        double      Ratio_Tolerance;

        cv::Mat H;                              // 2x6
        cv::Mat R;                              // 2x2

        bool        Has_Reference = false;
        std::vector<cv::Point2d> Reference_Local_Coords;
        double      Reference_Ratio = 0.0;

        bool        Initialized = false;
        std::int64_t Last_Timestamp = 0;
        std::vector<TrackedLED> LEDs;

    public:
        Kalman_LED_Tracker(cv::Size Resolution,
                           int      Target_Cluster_Count,
                           double   Process_Noise_Sigma     = 500000.0,
                           double   Measurement_Noise_Sigma = 3.0,
                           double   Gate_Sigma              = 3.0,
                           int      Min_Gate_Half_Size      = 20,
                           int      Max_Gate_Half_Size      = 120,
                           int      Min_Hot_Pixels          = 30,
                           int      Lost_Threshold          = 3,
                           int      Lost_Grace_Frames       = 15,
                           double   Expected_Ratio          = 0.0,
                           double   Ratio_Tolerance         = 0.3);

        void Update(const std::vector<cv::Point2i>& Hot_Pixels, std::int64_t Timestamp);
        dv::EventStore Get_Events_In_Gate(int LED_Index, const dv::EventStore& Events) const;
        const std::vector<TrackedLED>& Get_LEDs() const;
        bool Is_Initialized() const;
        void Reset();
        void Draw(cv::Mat& Frame, cv::Scalar Gate_Color = cv::Scalar(0, 255, 0), cv::Scalar Velocity_Color = cv::Scalar(255, 0, 0)) const;
        cv::Rect Get_Gate_Rect(int LED_Index) const;

    private:
        bool Initialize(const std::vector<cv::Point2i>& Hot_Pixels);
        void Track(const std::vector<cv::Point2i>& Hot_Pixels, double dt);
        void Build_Prediction_Matrices(double dt, cv::Mat& F, cv::Mat& Q) const;

        struct PCA_Result {
            cv::Point2d Mean;
            double Eigenvalues[2];
            double Eigenvectors[2][2];
            double Ratio;
            std::vector<cv::Point2d> Local_Coords;
        };
        PCA_Result Compute_PCA(const cv::Mat& Centers) const;
        std::vector<int> Match_Identities(const PCA_Result& PCA) const;
        static cv::Point2d Compute_Centroid(const std::vector<cv::Point2i>& Points);
};
