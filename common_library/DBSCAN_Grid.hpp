#pragma once

#include <vector>
#include <cstdint>
#include <dv-processing/core/core.hpp>
#include <opencv2/core.hpp>

class DBSCAN_Grid  {
    /* Private Data Members */
    private:
        std::int32_t Epsilon;                   // Radius in pixels
        std::int32_t Epsilon_Squared;           // Avoids sqrt in Range_Query() for quicker computation
        std::int32_t Minimum_Points;            // Minimum points to be declared a cluster
        cv::Size Resolution;                    // Height and Width of the grid
        std::vector<std::int32_t> Pixel_States;   // 1-D grid that stores the cluster labels and visit status for every pixel
        std::vector<cv::Point2i> Points;        // The points to cluster together

    /* Public Defined Structures */
    public:
        struct ClusterResult {
            std::vector<cv::Point2i> Centers;    // Centroid of each cluster
            std::vector<std::int32_t> Labels;    // Label per event (-1 = noise)
            std::vector<std::int32_t> Counts;    // Number of points per cluster
        };

    /* Public Functions */
    public:
        /**
         * Constructs a DBSCAN_Grid Object given a std::vector container of cv::Pointi, camera resolution, Epsilon or search radius, and the minimum number of points per cluster.
         */
        DBSCAN_Grid(const std::vector<cv::Point2i>& Pixels, const cv::Size Resolution, int Epsilon, int Minimum_Points);

        /**
         * Constructs a DBSCAN_Grid Object given the dv::EventStore container, camera resolution, Epsilon or search radius, and the minimum number of points per cluster.
         * Designed to be used with other dv::processing libraries.
         */
        DBSCAN_Grid(const dv::EventStore& Events, const cv::Size Resolution, int Epsilon, int Minimum_Points);

        /**
         * Begins the clustering algorithm after construction. Returns 3 vectors
         */
        ClusterResult Fit();

    /* Private Helper Functions */
    private:
        void Range_Query(int Points_Index, std::vector<std::int32_t>& Neighbors);
        void Expand_Cluster(std::vector<std::int32_t>& Seeds, std::vector<std::int32_t>& Scratch, std::vector<std::int32_t>& Labels, std::int32_t Cluster_ID);
};