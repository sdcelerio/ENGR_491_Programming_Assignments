#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <dv-processing/core/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "Multi_Car_Tracker.hpp"


Multi_Car_Tracker::Multi_Car_Tracker(int Num_Cars, int LEDs_Per_Car, cv::Size Resolution,
                                     double Process_Noise, double Meas_Noise)
    : Num_Cars(Num_Cars),
      LEDs_Per_Car(LEDs_Per_Car),
      Total_LEDs(Num_Cars * LEDs_Per_Car),
      Tracker(Resolution, Num_Cars * LEDs_Per_Car,
              Process_Noise, Meas_Noise,
              3.0,    // gate sigma
              20,     // min gate
              120,    // max gate
              Total_LEDs * 3,  // min hot pixels (3 per LED minimum)
              8,      // lost threshold
              5,      // grace frames
              0.0,    // expected ratio (skip)
              0.3)    // ratio tolerance
{
    this->Cars.resize(Num_Cars);
    for (int i = 0; i < Num_Cars; ++i)
        this->Cars[i].LED_Indices.resize(LEDs_Per_Car);
}


void Multi_Car_Tracker::Update(const std::vector<cv::Point2i>& Hot_Pixels, std::int64_t Timestamp) {
    bool Was_Initialized = this->Tracker.Is_Initialized();

    this->Tracker.Update(Hot_Pixels, Timestamp);

    if (!this->Tracker.Is_Initialized()) {
        this->Grouped = false;
        return;
    }

    // If tracker just initialized or re-initialized, group LEDs into cars
    if (!Was_Initialized && this->Tracker.Is_Initialized()) {
        this->Group_LEDs();
    }

    // If tracker reset (was grouped but tracker lost init), clear grouping
    if (this->Grouped) {
        this->Update_Car_States();
    }
}

dv::EventStore Multi_Car_Tracker::Get_Events_For_Car(int Car_Index, const dv::EventStore& Events) const {
    dv::EventStore Combined;
    if (Car_Index < 0 || Car_Index >= this->Num_Cars || !this->Grouped)
        return Combined;

    // Single pass through events — check if each event falls in any of this car's gates
    // This preserves timestamp order naturally
    std::vector<cv::Rect> Gates;
    for (int led : this->Cars[Car_Index].LED_Indices) {
        if (this->Tracker.Get_LEDs()[led].State_Flag == Kalman_LED_Tracker::Status::Tracking)
            Gates.push_back(this->Tracker.Get_Gate_Rect(led));
    }

    for (const dv::Event& E : Events) {
        cv::Point2i Pt(E.x(), E.y());
        for (const cv::Rect& Gate : Gates) {
            if (Gate.contains(Pt)) {
                Combined.emplace_back(E.timestamp(), E.x(), E.y(), E.polarity());
                break;  // Don't add same event twice if gates overlap
            }
        }
    }
    return Combined;
}

const std::vector<Multi_Car_Tracker::CarState>& Multi_Car_Tracker::Get_Cars() const {
    return this->Cars;
}

const Kalman_LED_Tracker& Multi_Car_Tracker::Get_Tracker() const {
    return this->Tracker;
}

bool Multi_Car_Tracker::Is_Initialized() const {
    return this->Tracker.Is_Initialized() && this->Grouped;
}

void Multi_Car_Tracker::Draw(cv::Mat& Frame) const {
    if (!this->Is_Initialized()) {
        cv::putText(Frame, "Initializing...",
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 0, 255), 1);
        return;
    }

    // Different colors per car
    cv::Scalar Car_Colors[] = {
        cv::Scalar(0, 255, 0),    // Car 0: green
        cv::Scalar(0, 165, 255),  // Car 1: orange
        cv::Scalar(255, 0, 0),    // Car 2: blue (if needed)
        cv::Scalar(255, 0, 255)   // Car 3: magenta (if needed)
    };
    cv::Scalar Vel_Colors[] = {
        cv::Scalar(0, 200, 0),
        cv::Scalar(0, 130, 200),
        cv::Scalar(200, 0, 0),
        cv::Scalar(200, 0, 200)
    };

    const auto& LEDs = this->Tracker.Get_LEDs();

    for (int c = 0; c < this->Num_Cars; ++c) {
        cv::Scalar Color = Car_Colors[c % 4];
        cv::Scalar VColor = Vel_Colors[c % 4];

        // Draw each LED in this car's color
        for (int led : this->Cars[c].LED_Indices) {
            if (LEDs[led].State_Flag == Kalman_LED_Tracker::Status::Lost)
                continue;

            double px = LEDs[led].State.at<double>(0);
            double py = LEDs[led].State.at<double>(1);
            double vx = LEDs[led].State.at<double>(2);
            double vy = LEDs[led].State.at<double>(3);
            cv::Point Center(static_cast<int>(px), static_cast<int>(py));

            // Gate rect
            cv::Rect Gate = this->Tracker.Get_Gate_Rect(led);
            cv::rectangle(Frame, Gate, Color, 1);

            // Centroid
            cv::circle(Frame, Center, 3, Color, -1);

            // Velocity
            cv::Point Vel_End(static_cast<int>(px + vx * 0.3), static_cast<int>(py + vy * 0.3));
            cv::arrowedLine(Frame, Center, Vel_End, VColor, 1);
        }

        // Draw car label at car centroid
        if (this->Cars[c].Tracking) {
            cv::Point Car_Center(
                static_cast<int>(this->Cars[c].Centroid.x),
                static_cast<int>(this->Cars[c].Centroid.y)
            );
            std::string Label = "Car " + std::to_string(c);
            cv::putText(Frame, Label,
                        cv::Point(Car_Center.x - 15, Car_Center.y - 25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, Color, 2);
        }
    }
}


/* ──────────────── Grouping ──────────────── */
void Multi_Car_Tracker::Group_LEDs() {
    const auto& LEDs = this->Tracker.Get_LEDs();
    int N = this->Total_LEDs;

    // Special case: 1 car — all LEDs belong to it
    if (this->Num_Cars == 1) {
        this->Cars[0].LED_Indices.resize(N);
        for (int i = 0; i < N; ++i)
            this->Cars[0].LED_Indices[i] = i;
        this->Grouped = true;
        this->Update_Car_States();
        std::cerr << "GROUPED: Car 0 LEDs=[";
        for (int i = 0; i < N; ++i) { if (i > 0) std::cerr << ","; std::cerr << i; }
        std::cerr << "]" << std::endl;
        return;
    }

    // Get all LED centers
    std::vector<cv::Point2d> Centers(N);
    for (int i = 0; i < N; ++i) {
        Centers[i].x = LEDs[i].State.at<double>(0);
        Centers[i].y = LEDs[i].State.at<double>(1);
    }

    // For 2 cars of 3 LEDs: brute force all C(6,3)/2 = 10 groupings
    // Generate all combinations of 3 indices from [0..5] for the first group
    // The second group is the remaining 3
    double Best_Cost = std::numeric_limits<double>::max();
    std::vector<int> Best_Group_A, Best_Group_B;

    std::vector<int> Indices(N);
    std::iota(Indices.begin(), Indices.end(), 0);

    // Generate all C(N, LEDs_Per_Car) combinations for group A
    std::vector<bool> Selector(N, false);
    std::fill(Selector.begin(), Selector.begin() + this->LEDs_Per_Car, true);

    // Use reverse iteration to generate combinations via prev_permutation
    std::sort(Selector.begin(), Selector.end(), std::greater<bool>());

    do {
        std::vector<int> Group_A, Group_B;
        for (int i = 0; i < N; ++i) {
            if (Selector[i]) Group_A.push_back(i);
            else Group_B.push_back(i);
        }

        // Skip duplicate groupings (A,B) == (B,A)
        // Only consider groupings where the smallest index is in Group A
        if (Group_A[0] > Group_B[0]) continue;

        // Compute within-group compactness: sum of all pairwise distances
        auto Group_Cost = [&](const std::vector<int>& Group) {
            double cost = 0.0;
            for (int i = 0; i < static_cast<int>(Group.size()); ++i) {
                for (int j = i + 1; j < static_cast<int>(Group.size()); ++j) {
                    double dx = Centers[Group[i]].x - Centers[Group[j]].x;
                    double dy = Centers[Group[i]].y - Centers[Group[j]].y;
                    cost += std::sqrt(dx * dx + dy * dy);
                }
            }
            return cost;
        };

        double Total_Cost = Group_Cost(Group_A) + Group_Cost(Group_B);

        if (Total_Cost < Best_Cost) {
            Best_Cost = Total_Cost;
            Best_Group_A = Group_A;
            Best_Group_B = Group_B;
        }

    } while (std::prev_permutation(Selector.begin(), Selector.end()));

    // Determine which group is "car 0" vs "car 1"
    // If we have previous car positions, match by nearest centroid
    // Otherwise, leftmost group = car 0
    auto Group_Centroid = [&](const std::vector<int>& Group) {
        cv::Point2d C(0, 0);
        for (int idx : Group) { C.x += Centers[idx].x; C.y += Centers[idx].y; }
        C.x /= Group.size(); C.y /= Group.size();
        return C;
    };

    cv::Point2d Centroid_A = Group_Centroid(Best_Group_A);
    cv::Point2d Centroid_B = Group_Centroid(Best_Group_B);

    if (this->Cars[0].Tracking && this->Cars[1].Tracking) {
        // Match to previous car positions
        double d00 = cv::norm(Centroid_A - this->Cars[0].Centroid);
        double d01 = cv::norm(Centroid_A - this->Cars[1].Centroid);
        if (d01 < d00) {
            std::swap(Best_Group_A, Best_Group_B);
        }
    } else {
        // First time: leftmost = car 0
        if (Centroid_A.x > Centroid_B.x) {
            std::swap(Best_Group_A, Best_Group_B);
        }
    }

    // Assign
    this->Cars[0].LED_Indices = Best_Group_A;
    this->Cars[1].LED_Indices = Best_Group_B;
    this->Grouped = true;

    this->Update_Car_States();

    std::cerr << "GROUPED: Car 0 LEDs=[" << Best_Group_A[0] << "," << Best_Group_A[1] << "," << Best_Group_A[2]
              << "] Car 1 LEDs=[" << Best_Group_B[0] << "," << Best_Group_B[1] << "," << Best_Group_B[2] << "]" << std::endl;
}


void Multi_Car_Tracker::Update_Car_States() {
    const auto& LEDs = this->Tracker.Get_LEDs();

    for (int c = 0; c < this->Num_Cars; ++c) {
        double sx = 0, sy = 0, svx = 0, svy = 0;
        int count = 0;

        for (int led : this->Cars[c].LED_Indices) {
            if (LEDs[led].State_Flag == Kalman_LED_Tracker::Status::Tracking) {
                sx += LEDs[led].State.at<double>(0);
                sy += LEDs[led].State.at<double>(1);
                svx += LEDs[led].State.at<double>(2);
                svy += LEDs[led].State.at<double>(3);
                count++;
            }
        }

        if (count > 0) {
            this->Cars[c].Centroid = {sx / count, sy / count};
            this->Cars[c].Velocity = {svx / count, svy / count};
            this->Cars[c].Tracking = true;
        } else {
            this->Cars[c].Tracking = false;
        }
    }
}
