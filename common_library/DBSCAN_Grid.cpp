#include <vector>
#include <cstdint>
#include <algorithm>
#include <dv-processing/core/core.hpp>
#include <opencv2/core.hpp>
#include "DBSCAN_Grid.hpp"


static constexpr std::int32_t PIXEL_EMPTY = -4;

static constexpr std::int32_t LABEL_UNVISITED = -3;   // not yet touched by Fit()
static constexpr std::int32_t LABEL_IN_QUEUE = -2;   // confirmed
static constexpr std::int32_t LABEL_NOISE = -1;   // confirmed

/* Constructors */
DBSCAN_Grid::DBSCAN_Grid(const dv::EventStore& Events, const cv::Size Resolution, int Epsilon, int Minimum_Points)
    : Epsilon(Epsilon),
      Epsilon_Squared(Epsilon * Epsilon),
      Minimum_Points(Minimum_Points),
      Resolution(Resolution),
      Pixel_States(Resolution.width * Resolution.height, PIXEL_EMPTY) {

    // Add the given events' coordinates into the Points vectors 
    this->Points.reserve(Events.size());
    for (const dv::Event& Event : Events) {
        this->Pixel_States[Event.y() * this->Resolution.width + Event.x()] = this->Points.size();
        this->Points.emplace_back(Event.x(), Event.y());
    }
}

DBSCAN_Grid::DBSCAN_Grid(const std::vector<cv::Point2i>& Pixels, const cv::Size Resolution, int Epsilon, int Minimum_Points)
    : Epsilon(Epsilon),
      Epsilon_Squared(Epsilon * Epsilon),
      Minimum_Points(Minimum_Points),
      Resolution(Resolution),
      Pixel_States(Resolution.width * Resolution.height, PIXEL_EMPTY) {
    
    // Add the given Pixels' into the Points vectors
    this->Points.reserve(Pixels.size());
    for (const cv::Point2i& Pixel : Pixels) {
        this->Pixel_States[Pixel.y * this->Resolution.width + Pixel.x] = this->Points.size();
        this->Points.emplace_back(Pixel.x, Pixel.y);
    }
}

/* Public functions */
DBSCAN_Grid::ClusterResult DBSCAN_Grid::Fit() {
    // Create a ClusterResult variable to return and append during the algorithm. Consider every point unvisted
    DBSCAN_Grid::ClusterResult Result;
    Result.Labels = std::vector<std::int32_t>(this->Points.size(), LABEL_UNVISITED);
    
    // Operate through the given set of points through the Points array
    std::int32_t Cluster_ID = 0;         // Start with an id of 0
    std::vector<std::int32_t> Neighbors; // Reseted every loop and stack defined to increase cache efficiency by making sure it never reallocates on every iteration
    std::vector<std::int32_t> Seeds;     // Expand_Cluster frontier
    Neighbors.reserve(256);
    Seeds.reserve(256);
    for (int Points_Index = 0; Points_Index < (int) this->Points.size(); ++Points_Index) {
        // If the point has already been visited then skip it
        if (Result.Labels[Points_Index] != LABEL_UNVISITED) 
            continue;

        // Get the neighbors of the current point and see if there is enough neighbors to be considered a core point of a cluster
        this->Range_Query(Points_Index, Neighbors);
        if ((std::int32_t) Neighbors.size() < this->Minimum_Points) {
            Result.Labels[Points_Index] = LABEL_NOISE;
            continue;
        }

        // Start a new cluster and attempt to expand from there by feeding unvisited points 
        Result.Labels[Points_Index] = Cluster_ID;
        Seeds.clear();
        for (const std::int32_t Neighbor : Neighbors) {
            if (Result.Labels[Neighbor] == LABEL_UNVISITED) {
                Result.Labels[Neighbor] = LABEL_IN_QUEUE;
                Seeds.push_back(Neighbor);
            }
        }

        Expand_Cluster(Seeds, Neighbors, Result.Labels, Cluster_ID);
        Cluster_ID++;
    }

    // After clustering points get the centers and counts per cluster
    Result.Centers.resize(Cluster_ID);
    Result.Counts.assign(Cluster_ID, 0);
    std::vector<std::int64_t> Sum_X(Cluster_ID, 0);
    std::vector<std::int64_t> Sum_Y(Cluster_ID, 0);
    for (int Points_Index = 0; Points_Index < (int) this->Points.size(); ++Points_Index) {
        // Skip any points that were considered noise or other labels
        const std::int32_t Cluster_Index = Result.Labels[Points_Index];
        if (Cluster_Index < 0) 
            continue; 

        Result.Counts[Cluster_Index]++;
        Sum_X[Cluster_Index] += this->Points[Points_Index].x;
        Sum_Y[Cluster_Index] += this->Points[Points_Index].y;
    }
    for (int c = 0; c < Cluster_ID; ++c) {
        Result.Centers[c] = {
            static_cast<int>(Sum_X[c] / Result.Counts[c]),
            static_cast<int>(Sum_Y[c] / Result.Counts[c])
        };
    }

    return Result;
}
void DBSCAN_Grid::Range_Query(int Points_Index, std::vector<std::int32_t>& Neighbors) {
    // Clear the neighbors vector and determine the search radius around the target
    Neighbors.clear();
    const cv::Point2i& Target_Point = this->Points[Points_Index];
    const int X_Min = std::max(0, Target_Point.x - this->Epsilon);
    const int X_Max = std::min(this->Resolution.width  - 1, Target_Point.x + this->Epsilon);
    const int Y_Min = std::max(0, Target_Point.y - this->Epsilon);
    const int Y_Max = std::min(this->Resolution.height - 1, Target_Point.y + this->Epsilon);

    // Search through the pixel states vector and see 
    for (int Y = Y_Min; Y <= Y_Max; ++Y) {
        const std::int32_t* Row = this->Pixel_States.data() + Y * this->Resolution.width;
        for (int X = X_Min; X <= X_Max; ++X) {
            // Skip any empty pixels
            if (Row[X] == PIXEL_EMPTY) 
                continue;

            const int dx = Target_Point.x - X;
            const int dy = Target_Point.y - Y;
            if (dx * dx + dy * dy <= this->Epsilon_Squared)
                Neighbors.push_back(Row[X]);
        }
    }
}

void DBSCAN_Grid::Expand_Cluster(std::vector<std::int32_t>& Seeds, std::vector<std::int32_t>& Neighbors, std::vector<std::int32_t>& Labels, std::int32_t Cluster_ID) {
    for (int i = 0; i < static_cast<int>(Seeds.size()); ++i) {
        const std::int32_t q = Seeds[i];

        // Check if the Seed was considered a noise but is now within the border. Prevent expanding if so
        if (Labels[q] == LABEL_NOISE) {
            Labels[q] = Cluster_ID; 
            continue;
        }

        Labels[q] = Cluster_ID;
        Range_Query(q, Neighbors);
        if ((std::int32_t) Neighbors.size() < this->Minimum_Points) 
            continue;

        for (const std::int32_t Neighbor : Neighbors) {
            if (Labels[Neighbor] == LABEL_UNVISITED) {
                Labels[Neighbor] = LABEL_IN_QUEUE;
                Seeds.push_back(Neighbor);
            } else if (Labels[Neighbor] == LABEL_NOISE) {
                Seeds.push_back(Neighbor);
            }
        }
    }
}