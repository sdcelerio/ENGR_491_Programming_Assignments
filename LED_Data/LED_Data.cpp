#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <dv-processing/core/core.hpp>
#include <dv-processing/io/camera/discovery.hpp>
#include <dv-processing/visualization/event_visualizer.hpp>
#include <opencv2/opencv.hpp>
#include "Hot_Pixel_Detector.hpp"
#include "Kalman_LED_Tracker.hpp"

/**
 * Measures the median frequency from raw events inside a gate.
 * Tracks the last positive-polarity timestamp per pixel.
 * Returns 0 if not enough measurements.
 */
struct Frequency_Estimator {
    int Width;
    std::vector<std::int64_t> Last_Pos_Timestamp;  // Per-pixel last positive event timestamp
    double Measured_Frequency = 0.0;

    Frequency_Estimator(cv::Size Resolution) 
        : Width(Resolution.width), 
          Last_Pos_Timestamp(Resolution.width * Resolution.height, 0) {}

    double Estimate(const dv::EventStore& Events) {
        if (Events.isEmpty()) return this->Measured_Frequency;

        std::vector<double> Frequencies;
        Frequencies.reserve(Events.size() / 2);

        for (const dv::Event& Event : Events) {
            // Only measure on positive polarity (OFF→ON = rising edge of blink)
            if (!Event.polarity()) continue;

            int Index = Event.y() * this->Width + Event.x();
            std::int64_t Prev = this->Last_Pos_Timestamp[Index];
            this->Last_Pos_Timestamp[Index] = Event.timestamp();

            // Skip if no previous timestamp for this pixel
            if (Prev == 0) continue;

            std::int64_t dt_us = Event.timestamp() - Prev;
            // Filter out unreasonable intervals (< 100 Hz or > 10000 Hz)
            if (dt_us > 100 && dt_us < 10000000) {
                double freq = 1e6 / static_cast<double>(dt_us);
                if (freq >= 50.0 && freq <= 5000.0)
                    Frequencies.push_back(freq);
            }
        }

        if (Frequencies.size() >= 5) {
            // Median frequency — robust to outliers
            std::sort(Frequencies.begin(), Frequencies.end());
            this->Measured_Frequency = Frequencies[Frequencies.size() / 2];
        }

        return this->Measured_Frequency;
    }
};

int main(void) {
    std::cerr << "Opening camera..." << std::endl;
    dv::io::camera::CameraPtr Camera = dv::io::camera::open();

    auto resolution = Camera->getEventResolution();
    if (!resolution.has_value()) {
        std::cerr << "Camera does not provide event resolution!" << std::endl;
        return 1;
    }
    std::cerr << "Resolution: " << resolution->width << "x" << resolution->height << std::endl;

    if (!Camera->isEventStreamAvailable()) {
        std::cerr << "Camera does not provide event stream!" << std::endl;
        return 1;
    }

    // ── Stage 1: Hot pixel detection ──
    Hot_Pixel_Detector Detector(*resolution, 3);

    // ── Stage 2: Kalman LED Tracker ──
    // Parameters:                  res,  N, proc_noise, meas_noise, gate_σ, min_gate, max_gate, min_px, lost, grace, ratio, tol
    Kalman_LED_Tracker Tracker(*resolution, 3, 500000.0,  3.0,        3.0,    20,       120,      30,     8,    5,     0.0,   0.3);

    // ── Stage 3: Per-LED frequency estimation ──
    std::vector<Frequency_Estimator> Freq_Estimators;
    for (int i = 0; i < 3; ++i)
        Freq_Estimators.emplace_back(*resolution);

    // ── Visualization ──
    dv::visualization::EventVisualizer Visualizer(*resolution,
        dv::visualization::colors::black,
        dv::visualization::colors::green,
        dv::visualization::colors::red);
    cv::namedWindow("Events", cv::WINDOW_NORMAL);
    cv::namedWindow("Tracking", cv::WINDOW_NORMAL);

    // Optional: record output
    // cv::VideoWriter Writer("tracking_output.avi",
    //     cv::VideoWriter::fourcc('M','J','P','G'), 30, *resolution);

    std::cerr << "Entering main loop..." << std::endl;

    while (Camera->isRunning()) {
        if (std::optional<dv::EventStore> Events = Camera->getNextEventBatch()) {
            if (!Events->isEmpty()) {
                // Stage 1: Hot pixels
                Detector.Process_Batch(*Events);
                const std::vector<cv::Point2i>& Hot_Pixels = Detector.Get_Hot_Pixels();

                // Stage 2: Kalman tracking
                std::int64_t Timestamp = Events->getHighestTime();
                if (!Hot_Pixels.empty())
                    Tracker.Update(Hot_Pixels, Timestamp);

                // Stage 3: Per-LED frequency estimation
                double LED_Frequencies[3] = {0.0, 0.0, 0.0};
                if (Tracker.Is_Initialized()) {
                    for (int i = 0; i < 3; ++i) {
                        dv::EventStore Gate_Events = Tracker.Get_Events_In_Gate(i, *Events);
                        LED_Frequencies[i] = Freq_Estimators[i].Estimate(Gate_Events);
                    }
                }

                // ── Draw ──
                cv::Mat Tracking_Frame(*resolution, CV_8UC3, cv::Vec3b(0, 0, 0));
                Detector.Highlight_Pixels(Tracking_Frame, cv::Vec3b(60, 60, 60));

                if (Tracker.Is_Initialized()) {
                    Tracker.Draw(Tracking_Frame);

                    const auto& LEDs = Tracker.Get_LEDs();
                    cv::Scalar Colors[] = {cv::Scalar(100, 100, 255), cv::Scalar(255, 100, 100), cv::Scalar(100, 255, 100)};
                    for (int i = 0; i < 3; ++i) {
                        if (LEDs[i].State_Flag == Kalman_LED_Tracker::Status::Lost)
                            continue;

                        double px = LEDs[i].State.at<double>(0);
                        double py = LEDs[i].State.at<double>(1);

                        // Display measured frequency
                        std::string Freq_Label;
                        if (LED_Frequencies[i] > 1.0) {
                            char buf[64];
                            if (LED_Frequencies[i] > 350 && LED_Frequencies[i] < 450)
                                std::snprintf(buf, sizeof(buf), "%.0f Hz IDLE", LED_Frequencies[i]);
                            else if (LED_Frequencies[i] > 950 && LED_Frequencies[i] < 1050)
                                std::snprintf(buf, sizeof(buf), "%.0f Hz SEND", LED_Frequencies[i]);
                            else if (LED_Frequencies[i] > 1150 && LED_Frequencies[i] < 1250)
                                std::snprintf(buf, sizeof(buf), "%.0f Hz STOP", LED_Frequencies[i]);
                            else if (LED_Frequencies[i] > 450 && LED_Frequencies[i] < 550)
                                std::snprintf(buf, sizeof(buf), "%.0f Hz 00", LED_Frequencies[i]);
                            else if (LED_Frequencies[i] > 550 && LED_Frequencies[i] < 650)
                                std::snprintf(buf, sizeof(buf), "%.0f Hz 01", LED_Frequencies[i]);
                            else if (LED_Frequencies[i] > 650 && LED_Frequencies[i] < 750)
                                std::snprintf(buf, sizeof(buf), "%.0f Hz 10", LED_Frequencies[i]);
                            else if (LED_Frequencies[i] > 750 && LED_Frequencies[i] < 850)
                                std::snprintf(buf, sizeof(buf), "%.0f Hz 11", LED_Frequencies[i]);
                            else
                                std::snprintf(buf, sizeof(buf), "%.0f Hz (ERROR)", LED_Frequencies[i]);
                            Freq_Label = buf;
                        } else {
                            Freq_Label = "measuring...";
                        }

                        cv::putText(Tracking_Frame, Freq_Label,
                                    cv::Point(static_cast<int>(px) - 30, static_cast<int>(py) + 30),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.4, Colors[i], 1);
                    }
                } else {
                    cv::putText(Tracking_Frame, "Initializing...",
                                cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX,
                                0.5, cv::Scalar(0, 0, 255), 1);
                }

                // Writer.write(Tracking_Frame);  // Uncomment to record
                cv::imshow("Events", Visualizer.generateImage(*Events));
                cv::imshow("Tracking", Tracking_Frame);
            }
        }
        cv::pollKey();
    }

    return 0;
}
