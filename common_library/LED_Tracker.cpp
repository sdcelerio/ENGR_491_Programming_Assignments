#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "LED_Tracker.hpp"
#include "DBSCAN_Grid.hpp"


/* Constructor */
LED_Tracker::LED_Tracker(cv::Size Resolution,
                         int      Target_Cluster_Count,
                         int      Box_Padding,
                         int      Epsilon_Min,
                         int      Epsilon_Max,
                         int      Min_Cluster_Size,
                         int      Lost_Threshold,
                         double   Velocity_Smoothing,
                         int      Max_Search_Iterations)
    : Resolution(Resolution),
      Target_Cluster_Count(Target_Cluster_Count),
      Box_Padding(Box_Padding),
      Epsilon_Min(Epsilon_Min),
      Epsilon_Max(Epsilon_Max),
      Min_Cluster_Size(Min_Cluster_Size),
      Lost_Threshold(Lost_Threshold),
      Velocity_Smoothing(Velocity_Smoothing),
      Max_Search_Iterations(Max_Search_Iterations) {

    this->LEDs.resize(Target_Cluster_Count);
}


/* Public Functions */
void LED_Tracker::Update(const std::vector<cv::Point2i>& Valid_Pixels) {
    // If not yet initialized, or a reset was requested, attempt initialization
    if (!this->Initialized) {
        this->Initialized = this->Initialize(Valid_Pixels);
        return;
    }

    // Otherwise run the lightweight per-box tracking
    this->Track(Valid_Pixels);
}

const std::vector<LED_Tracker::TrackedLED>& LED_Tracker::Get_LEDs() const {
    return this->LEDs;
}

bool LED_Tracker::Is_Initialized() const {
    return this->Initialized;
}

void LED_Tracker::Reset() {
    this->Initialized = false;
    for (TrackedLED& LED : this->LEDs) {
        LED.State       = Status::Lost;
        LED.Event_Count = 0;
        LED.Velocity    = {0.0, 0.0};
    }
}

void LED_Tracker::Draw(cv::Mat& Frame, cv::Scalar Box_Color, cv::Scalar Velocity_Color) const {
    for (int i = 0; i < static_cast<int>(this->LEDs.size()); ++i) {
        const TrackedLED& LED = this->LEDs[i];
        if (LED.State == Status::Lost)
            continue;

        // Draw bounding box
        cv::rectangle(Frame, LED.Bounding_Box, Box_Color, 1);

        // Draw centroid as a small filled circle
        cv::Point Center(static_cast<int>(LED.Centroid.x), static_cast<int>(LED.Centroid.y));
        cv::circle(Frame, Center, 3, Box_Color, -1);

        // Draw velocity vector (scaled up for visibility)
        constexpr double Velocity_Scale = 50.0;
        cv::Point Velocity_End(
            static_cast<int>(LED.Centroid.x + LED.Velocity.x * Velocity_Scale),
            static_cast<int>(LED.Centroid.y + LED.Velocity.y * Velocity_Scale)
        );
        cv::arrowedLine(Frame, Center, Velocity_End, Velocity_Color, 1);

        // Label with LED index and event count
        std::string Label = "LED " + std::to_string(i) + " [" + std::to_string(LED.Event_Count) + "]";
        cv::putText(Frame, Label,
                    cv::Point(LED.Bounding_Box.x, LED.Bounding_Box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, Box_Color, 1);
    }
}


/* Private Helper Functions */
bool LED_Tracker::Initialize(const std::vector<cv::Point2i>& Valid_Pixels) {
    // Need a reasonable number of points to even attempt clustering
    if (static_cast<int>(Valid_Pixels.size()) < this->Target_Cluster_Count * this->Min_Cluster_Size)
        return false;

    // Binary search on epsilon to find the value that yields exactly Target_Cluster_Count valid clusters
    int Best_Epsilon     = -1;
    int Best_Cluster_Diff = std::numeric_limits<int>::max();
    DBSCAN_Grid::ClusterResult Best_Result;

    int Low  = this->Epsilon_Min;
    int High = this->Epsilon_Max;

    for (int Iteration = 0; Iteration < this->Max_Search_Iterations && Low <= High; ++Iteration) {
        int Mid = (Low + High) / 2;

        // Run DBSCAN with the candidate epsilon
        DBSCAN_Grid Clusterer(Valid_Pixels, this->Resolution, Mid, this->Min_Cluster_Size);
        DBSCAN_Grid::ClusterResult Result = Clusterer.Fit();

        // Count only clusters that meet the minimum size threshold
        int Valid_Cluster_Count = 0;
        for (int c = 0; c < static_cast<int>(Result.Counts.size()); ++c) {
            if (Result.Counts[c] >= this->Min_Cluster_Size)
                Valid_Cluster_Count++;
        }

        int Diff = std::abs(Valid_Cluster_Count - this->Target_Cluster_Count);
        if (Diff < Best_Cluster_Diff) {
            Best_Cluster_Diff = Diff;
            Best_Epsilon      = Mid;
            Best_Result       = Result;
        }

        // If we found the exact target, stop searching
        if (Valid_Cluster_Count == this->Target_Cluster_Count)
            break;

        // Too many clusters → epsilon is too small, increase it to merge clusters
        if (Valid_Cluster_Count > this->Target_Cluster_Count)
            Low = Mid + 1;
        // Too few clusters → epsilon is too large, decrease it to split clusters
        else
            High = Mid - 1;
    }

    // If we couldn't get close enough to the target, fail initialization
    if (Best_Cluster_Diff > 0)
        return false;

    // Build the tracked LED states from the best result, picking only valid-size clusters
    // Sort clusters by size descending so we pick the largest ones if there are extras
    std::vector<int> Valid_Indices;
    for (int c = 0; c < static_cast<int>(Best_Result.Counts.size()); ++c) {
        if (Best_Result.Counts[c] >= this->Min_Cluster_Size)
            Valid_Indices.push_back(c);
    }

    // Gather the points belonging to each valid cluster so we can build tight bounding boxes
    std::vector<std::vector<cv::Point2i>> Cluster_Points(Valid_Indices.size());
    for (int p = 0; p < static_cast<int>(Valid_Pixels.size()); ++p) {
        std::int32_t Label = Best_Result.Labels[p];
        if (Label < 0)
            continue;

        // Find which valid index this label maps to
        for (int v = 0; v < static_cast<int>(Valid_Indices.size()); ++v) {
            if (Valid_Indices[v] == Label) {
                Cluster_Points[v].push_back(Valid_Pixels[p]);
                break;
            }
        }
    }

    // Initialize each TrackedLED
    for (int i = 0; i < this->Target_Cluster_Count; ++i) {
        this->LEDs[i].Centroid     = Compute_Centroid(Cluster_Points[i]);
        this->LEDs[i].Velocity     = {0.0, 0.0};
        this->LEDs[i].Bounding_Box = Build_Bounding_Box(Cluster_Points[i]);
        this->LEDs[i].Event_Count  = static_cast<int>(Cluster_Points[i].size());
        this->LEDs[i].State        = Status::Tracking;
    }

    return true;
}

void LED_Tracker::Track(const std::vector<cv::Point2i>& Valid_Pixels) {
    int Lost_Count = 0;

    for (TrackedLED& LED : this->LEDs) {
        // Predict the new box center using current velocity
        cv::Point2d Predicted_Center(
            LED.Centroid.x + LED.Velocity.x,
            LED.Centroid.y + LED.Velocity.y
        );

        // Build the search region around the predicted center
        int Half_W = LED.Bounding_Box.width  / 2;
        int Half_H = LED.Bounding_Box.height / 2;
        cv::Rect Search_Region(
            static_cast<int>(Predicted_Center.x) - Half_W,
            static_cast<int>(Predicted_Center.y) - Half_H,
            LED.Bounding_Box.width,
            LED.Bounding_Box.height
        );

        // Clamp search region to resolution bounds
        Search_Region &= cv::Rect(0, 0, this->Resolution.width, this->Resolution.height);

        // Gather valid pixels that fall inside the search region
        std::vector<cv::Point2i> Pixels_In_Box;
        Pixels_In_Box.reserve(256);
        for (const cv::Point2i& Pixel : Valid_Pixels) {
            if (Search_Region.contains(Pixel))
                Pixels_In_Box.push_back(Pixel);
        }

        LED.Event_Count = static_cast<int>(Pixels_In_Box.size());

        // Check if the LED is lost
        if (LED.Event_Count < this->Lost_Threshold) {
            LED.State = Status::Lost;
            Lost_Count++;
            continue;
        }

        // Compute the measured centroid
        cv::Point2d Measured_Centroid = Compute_Centroid(Pixels_In_Box);

        // Update velocity with exponential moving average
        cv::Point2d Displacement(
            Measured_Centroid.x - LED.Centroid.x,
            Measured_Centroid.y - LED.Centroid.y
        );
        LED.Velocity.x = this->Velocity_Smoothing * Displacement.x + (1.0 - this->Velocity_Smoothing) * LED.Velocity.x;
        LED.Velocity.y = this->Velocity_Smoothing * Displacement.y + (1.0 - this->Velocity_Smoothing) * LED.Velocity.y;

        // Update centroid to measured position (correct the prediction)
        LED.Centroid = Measured_Centroid;

        // Re-center bounding box on the measured centroid
        LED.Bounding_Box.x = static_cast<int>(Measured_Centroid.x) - Half_W;
        LED.Bounding_Box.y = static_cast<int>(Measured_Centroid.y) - Half_H;

        // Clamp the bounding box to resolution
        LED.Bounding_Box &= cv::Rect(0, 0, this->Resolution.width, this->Resolution.height);

        LED.State = Status::Tracking;
    }

    // If too many LEDs are lost, trigger re-initialization
    if (Lost_Count > this->Target_Cluster_Count / 2)
        this->Reset();
}

cv::Rect LED_Tracker::Build_Bounding_Box(const std::vector<cv::Point2i>& Cluster_Points) const {
    // Find the axis-aligned extent of the cluster
    int X_Min = std::numeric_limits<int>::max();
    int X_Max = std::numeric_limits<int>::min();
    int Y_Min = std::numeric_limits<int>::max();
    int Y_Max = std::numeric_limits<int>::min();

    for (const cv::Point2i& Point : Cluster_Points) {
        X_Min = std::min(X_Min, Point.x);
        X_Max = std::max(X_Max, Point.x);
        Y_Min = std::min(Y_Min, Point.y);
        Y_Max = std::max(Y_Max, Point.y);
    }

    // Add padding and clamp to resolution
    X_Min = std::max(0, X_Min - this->Box_Padding);
    Y_Min = std::max(0, Y_Min - this->Box_Padding);
    X_Max = std::min(this->Resolution.width  - 1, X_Max + this->Box_Padding);
    Y_Max = std::min(this->Resolution.height - 1, Y_Max + this->Box_Padding);

    return cv::Rect(X_Min, Y_Min, X_Max - X_Min + 1, Y_Max - Y_Min + 1);
}

cv::Point2d LED_Tracker::Compute_Centroid(const std::vector<cv::Point2i>& Points) {
    double Sum_X = 0.0;
    double Sum_Y = 0.0;
    for (const cv::Point2i& Point : Points) {
        Sum_X += Point.x;
        Sum_Y += Point.y;
    }
    double N = static_cast<double>(Points.size());
    return {Sum_X / N, Sum_Y / N};
}
