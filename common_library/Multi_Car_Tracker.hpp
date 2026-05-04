#pragma once

#include <vector>
#include <cstdint>
#include <dv-processing/core/core.hpp>
#include <opencv2/core.hpp>
#include "Kalman_LED_Tracker.hpp"

class Multi_Car_Tracker {
    public:
        struct CarState {
            cv::Point2d Centroid;           // Mean position of this car's LEDs
            cv::Point2d Velocity;           // Mean velocity of this car's LEDs
            std::vector<int> LED_Indices;   // Which LED indices belong to this car
            bool Tracking = false;
        };

    private:
        int Num_Cars;
        int LEDs_Per_Car;
        int Total_LEDs;
        Kalman_LED_Tracker Tracker;

        std::vector<CarState> Cars;
        bool Grouped = false;

    public:
        /**
         * @param Num_Cars          Number of cars to track
         * @param LEDs_Per_Car      LEDs per car (e.g. 3)
         * @param Resolution        Camera resolution
         * @param Process_Noise     Kalman process noise sigma
         * @param Meas_Noise        Kalman measurement noise sigma
         */
        Multi_Car_Tracker(int Num_Cars, int LEDs_Per_Car, cv::Size Resolution,
                          double Process_Noise = 500000.0, double Meas_Noise = 3.0);

        /**
         * Main update call.
         */
        void Update(const std::vector<cv::Point2i>& Hot_Pixels, std::int64_t Timestamp);

        /**
         * Get events inside a specific car's combined gating region.
         */
        dv::EventStore Get_Events_For_Car(int Car_Index, const dv::EventStore& Events) const;

        const std::vector<CarState>& Get_Cars() const;
        const Kalman_LED_Tracker& Get_Tracker() const;
        bool Is_Initialized() const;

        /**
         * Draw all cars with different colors.
         */
        void Draw(cv::Mat& Frame) const;

    private:
        /**
         * Groups Total_LEDs LED centers into Num_Cars groups of LEDs_Per_Car
         * by finding the grouping with minimum total within-group distance.
         */
        void Group_LEDs();

        /**
         * Updates car centroids and velocities from their assigned LEDs.
         */
        void Update_Car_States();
};
