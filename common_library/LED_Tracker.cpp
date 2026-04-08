#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <dv-processing/core/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "LED_Tracker.hpp"


/* Constructor */
LED_Tracker::LED_Tracker(cv::Size Resolution,
                         int      Target_Cluster_Count,
                         int      Box_Half_Size,
                         int      Min_Hot_Pixels,
                         int      Lost_Threshold,
                         int      Lost_Grace_Frames,
                         double   Velocity_Smoothing,
                         double   Expected_Ratio,
                         double   Ratio_Tolerance)
    : Resolution(Resolution),
      Target_Cluster_Count(Target_Cluster_Count),
      Box_Half_Size(Box_Half_Size),
      Min_Hot_Pixels(Min_Hot_Pixels),
      Lost_Threshold(Lost_Threshold),
      Lost_Grace_Frames(Lost_Grace_Frames),
      Velocity_Smoothing(Velocity_Smoothing),
      Expected_Ratio(Expected_Ratio),
      Ratio_Tolerance(Ratio_Tolerance) {

    this->LEDs.resize(Target_Cluster_Count);
}


/* Public Functions */
void LED_Tracker::Update(const std::vector<cv::Point2i>& Hot_Pixels) {
    if (!this->Initialized) {
        this->Initialized = this->Initialize(Hot_Pixels);
        return;
    }
    this->Track(Hot_Pixels);
}

dv::EventStore LED_Tracker::Get_Events_In_Box(int LED_Index, const dv::EventStore& Events) const {
    dv::EventStore Filtered;
    if (LED_Index < 0 || LED_Index >= static_cast<int>(this->LEDs.size()))
        return Filtered;

    const TrackedLED& LED = this->LEDs[LED_Index];
    if (LED.State == Status::Lost)
        return Filtered;

    for (const dv::Event& Event : Events) {
        if (LED.Bounding_Box.contains(cv::Point2i(Event.x(), Event.y())))
            Filtered.emplace_back(Event.timestamp(), Event.x(), Event.y(), Event.polarity());
    }
    return Filtered;
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
        LED.Lost_Frames = 0;
        LED.Velocity    = {0.0, 0.0};
    }
}

void LED_Tracker::Draw(cv::Mat& Frame, cv::Scalar Box_Color, cv::Scalar Velocity_Color) const {
    for (int i = 0; i < static_cast<int>(this->LEDs.size()); ++i) {
        const TrackedLED& LED = this->LEDs[i];
        if (LED.State == Status::Lost)
            continue;

        cv::rectangle(Frame, LED.Bounding_Box, Box_Color, 1);

        cv::Point Center(static_cast<int>(LED.Centroid.x), static_cast<int>(LED.Centroid.y));
        cv::circle(Frame, Center, 3, Box_Color, -1);

        constexpr double Velocity_Scale = 50.0;
        cv::Point Velocity_End(
            static_cast<int>(LED.Centroid.x + LED.Velocity.x * Velocity_Scale),
            static_cast<int>(LED.Centroid.y + LED.Velocity.y * Velocity_Scale)
        );
        cv::arrowedLine(Frame, Center, Velocity_End, Velocity_Color, 1);

        std::string Label = "LED " + std::to_string(i) + " [" + std::to_string(LED.Event_Count) + "]";
        cv::putText(Frame, Label,
                    cv::Point(LED.Bounding_Box.x, LED.Bounding_Box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, Box_Color, 1);
    }
}


/* Private Helper Functions */
bool LED_Tracker::Initialize(const std::vector<cv::Point2i>& Hot_Pixels) {
    if (static_cast<int>(Hot_Pixels.size()) < this->Min_Hot_Pixels)
        return false;

    // Build the input matrix for k-means (N x 2, float)
    cv::Mat Points(static_cast<int>(Hot_Pixels.size()), 2, CV_32F);
    for (int i = 0; i < static_cast<int>(Hot_Pixels.size()); ++i) {
        Points.at<float>(i, 0) = static_cast<float>(Hot_Pixels[i].x);
        Points.at<float>(i, 1) = static_cast<float>(Hot_Pixels[i].y);
    }

    // Run k-means — guaranteed to return exactly Target_Cluster_Count clusters
    cv::Mat Labels, Centers;
    double Compactness = cv::kmeans(
        Points, this->Target_Cluster_Count, Labels,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 1.0),
        5, cv::KMEANS_PP_CENTERS, Centers
    );

    // Validate cluster geometry using eigenvalue ratio of centers
    if (!this->Validate_Geometry(Centers))
        return false;

    // Count events per cluster
    std::vector<int> Cluster_Counts(this->Target_Cluster_Count, 0);
    for (int i = 0; i < Labels.rows; ++i)
        Cluster_Counts[Labels.at<int>(i)]++;

    // Initialize each TrackedLED
    std::cerr << "INIT SUCCESS: " << Hot_Pixels.size() << " hot pixels" << std::endl;
    for (int i = 0; i < this->Target_Cluster_Count; ++i) {
        cv::Point2d Center(Centers.at<float>(i, 0), Centers.at<float>(i, 1));

        this->LEDs[i].Centroid     = Center;
        this->LEDs[i].Velocity     = {0.0, 0.0};
        this->LEDs[i].Bounding_Box = this->Build_Bounding_Box(Center);
        this->LEDs[i].Event_Count  = Cluster_Counts[i];
        this->LEDs[i].Lost_Frames  = 0;
        this->LEDs[i].State        = Status::Tracking;

        std::cerr << "  LED " << i << " center=(" << Center.x << "," << Center.y
                  << ") events=" << Cluster_Counts[i] << std::endl;
    }

    return true;
}

void LED_Tracker::Track(const std::vector<cv::Point2i>& Hot_Pixels) {
    int Lost_Count = 0;

    for (TrackedLED& LED : this->LEDs) {
        // Predict the new center using current velocity
        cv::Point2d Predicted_Center(
            LED.Centroid.x + LED.Velocity.x,
            LED.Centroid.y + LED.Velocity.y
        );

        // Build the fixed-size search region around the predicted center
        cv::Rect Search_Region = this->Build_Bounding_Box(Predicted_Center);

        // Gather hot pixels inside the search region
        std::vector<cv::Point2i> Pixels_In_Box;
        Pixels_In_Box.reserve(256);
        for (const cv::Point2i& Pixel : Hot_Pixels) {
            if (Search_Region.contains(Pixel))
                Pixels_In_Box.push_back(Pixel);
        }

        LED.Event_Count = static_cast<int>(Pixels_In_Box.size());

        if (LED.Event_Count < this->Lost_Threshold) {
            LED.Lost_Frames++;
            if (LED.Lost_Frames > this->Lost_Grace_Frames) {
                LED.State = Status::Lost;
                Lost_Count++;
            }
            continue;
        }

        LED.Lost_Frames = 0;

        // Compute the measured centroid
        cv::Point2d Measured_Centroid = Compute_Centroid(Pixels_In_Box);

        // Update velocity with exponential moving average
        cv::Point2d Displacement(
            Measured_Centroid.x - LED.Centroid.x,
            Measured_Centroid.y - LED.Centroid.y
        );
        LED.Velocity.x = this->Velocity_Smoothing * Displacement.x + (1.0 - this->Velocity_Smoothing) * LED.Velocity.x;
        LED.Velocity.y = this->Velocity_Smoothing * Displacement.y + (1.0 - this->Velocity_Smoothing) * LED.Velocity.y;

        // Update centroid to measured position
        LED.Centroid = Measured_Centroid;

        // Re-center the fixed-size box on the measured centroid
        LED.Bounding_Box = this->Build_Bounding_Box(Measured_Centroid);

        LED.State = Status::Tracking;
    }

    if (Lost_Count > 0) {
        std::cerr << "TRACKING RESET: " << Lost_Count << "/" << this->Target_Cluster_Count << " LEDs lost" << std::endl;
        this->Reset();
    }
}

cv::Rect LED_Tracker::Build_Bounding_Box(cv::Point2d Center) const {
    int X = static_cast<int>(Center.x) - this->Box_Half_Size;
    int Y = static_cast<int>(Center.y) - this->Box_Half_Size;
    int Size = this->Box_Half_Size * 2 + 1;
    cv::Rect Box(X, Y, Size, Size);
    Box &= cv::Rect(0, 0, this->Resolution.width, this->Resolution.height);
    return Box;
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

bool LED_Tracker::Validate_Geometry(const cv::Mat& Centers) const {
    // If Expected_Ratio is 0, skip the check (not configured yet)
    // Also print the measured ratio so the user can determine what to set it to
    int N = Centers.rows;

    // Need at least 3 centers for a meaningful eigenvalue ratio
    // For 2 centers, one eigenvalue is always 0 — fall back to minimum separation check
    if (N < 3) {
        // Simple separation check for 2 centers
        float dx = Centers.at<float>(0, 0) - Centers.at<float>(1, 0);
        float dy = Centers.at<float>(0, 1) - Centers.at<float>(1, 1);
        double Dist = std::sqrt(dx * dx + dy * dy);
        double Min_Sep = 2.0 * this->Box_Half_Size;
        if (Dist < Min_Sep) {
            std::cerr << "INIT REJECTED: 2 centers too close (dist=" << Dist 
                      << " min=" << Min_Sep << ")" << std::endl;
            return false;
        }
        return true;
    }

    // Compute the mean of all cluster centers
    double Mean_X = 0.0, Mean_Y = 0.0;
    for (int i = 0; i < N; ++i) {
        Mean_X += Centers.at<float>(i, 0);
        Mean_Y += Centers.at<float>(i, 1);
    }
    Mean_X /= N;
    Mean_Y /= N;

    // Compute the 2x2 covariance matrix of the cluster centers
    double Cov_XX = 0.0, Cov_XY = 0.0, Cov_YY = 0.0;
    for (int i = 0; i < N; ++i) {
        double dx = Centers.at<float>(i, 0) - Mean_X;
        double dy = Centers.at<float>(i, 1) - Mean_Y;
        Cov_XX += dx * dx;
        Cov_XY += dx * dy;
        Cov_YY += dy * dy;
    }
    Cov_XX /= (N - 1);
    Cov_XY /= (N - 1);
    Cov_YY /= (N - 1);

    // Eigenvalues of 2x2 covariance matrix via quadratic formula
    double b = -(Cov_XX + Cov_YY);
    double c = (Cov_XX * Cov_YY) - (Cov_XY * Cov_XY);
    double Discriminant = std::max(0.0, b * b - 4.0 * c);
    double Eigenvalue_1 = (-b + std::sqrt(Discriminant)) / 2.0;  // Larger
    double Eigenvalue_2 = (-b - std::sqrt(Discriminant)) / 2.0;  // Smaller

    // Compute ratio (larger / smaller). Guard against near-zero second eigenvalue
    // which means all centers are nearly collinear
    double Measured_Ratio = 0.0;
    if (Eigenvalue_2 > 1e-6)
        Measured_Ratio = Eigenvalue_1 / Eigenvalue_2;
    else
        Measured_Ratio = 1e6;  // Effectively infinite — collinear centers

    // Always print the measured ratio so user can determine the expected value
    std::cerr << "  Geometry: eigenvalues=(" << Eigenvalue_1 << ", " << Eigenvalue_2 
              << ") ratio=" << Measured_Ratio << std::endl;

    // If Expected_Ratio is 0, skip validation (user hasn't configured it yet)
    if (this->Expected_Ratio <= 0.0) {
        std::cerr << "  Ratio check SKIPPED (Expected_Ratio not set — set it to ~" 
                  << Measured_Ratio << " once LED geometry is finalized)" << std::endl;
        return true;
    }

    // Check if measured ratio is within tolerance of expected
    double Ratio_Error = std::abs(Measured_Ratio - this->Expected_Ratio) / this->Expected_Ratio;
    if (Ratio_Error > this->Ratio_Tolerance) {
        std::cerr << "  INIT REJECTED: ratio=" << Measured_Ratio 
                  << " expected=" << this->Expected_Ratio
                  << " error=" << (Ratio_Error * 100.0) << "%" 
                  << " tolerance=" << (this->Ratio_Tolerance * 100.0) << "%" << std::endl;
        return false;
    }

    std::cerr << "  Ratio check PASSED (error=" << (Ratio_Error * 100.0) << "%)" << std::endl;
    return true;
}
