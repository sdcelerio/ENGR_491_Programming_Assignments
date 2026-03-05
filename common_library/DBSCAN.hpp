#pragma once

#include <vector>
#include <cstdint>
#include <dv-processing/core/core.hpp>
#include <opencv2/core.hpp>

class DBSCAN_Grid  {
    /* Private Data Members */
    private:
        float Epsilon_Squared;
        int Minimum_Points;
        cv::Size Resolution;
        std::vector<int32_t> Pixel_Grid;
        std::vector<cv::Point2f> Points;

    /* Public Defined Structures */
    public:
        struct ClusterResult {
            std::vector<cv::Point2f> Centers;    // Centroid of each cluster
            std::vector<std::int32_t> Labels;    // Label per event (-1 = noise)
            std::vector<std::int32_t> Counts;    // Number of points per cluster
        };

    /* Public Functions */
    public:
        DBSCAN_Grid (const dv::EventStore& Events, const cv::Size Resolution, double Epsilon, int Minimum_Points);
        DBSCAN_Grid (const std::vector<cv::Point2i> Pixels, const cv::Size Resolution, double Epsilon, int Minimum_Points);
        ClusterResult Fit();

    /* Private Helper Functions */
    private:
        std::vector<std::int32_t> Range_Query(int Index);
        void Expand_Cluster(std::vector<const dv::Event*>& Neighbors);
};