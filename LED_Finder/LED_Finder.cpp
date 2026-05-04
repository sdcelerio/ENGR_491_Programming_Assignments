#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <dv-processing/core/core.hpp>
#include <dv-processing/io/camera/discovery.hpp>
#include <dv-processing/visualization/event_visualizer.hpp>
#include <opencv2/opencv.hpp>
#include "Hot_Pixel_Detector.hpp"
#include "Multi_Car_Tracker.hpp"

struct Frequency_Estimator {
    int Width;
    std::vector<std::int64_t> Last_Pos_Timestamp;
    double Measured_Frequency = 0.0;

    Frequency_Estimator(cv::Size Resolution)
        : Width(Resolution.width),
          Last_Pos_Timestamp(Resolution.width * Resolution.height, 0) {}

    double Estimate(const dv::EventStore& Events) {
        if (Events.isEmpty()) return this->Measured_Frequency;

        std::vector<double> Frequencies;
        Frequencies.reserve(Events.size() / 2);

        for (const dv::Event& Event : Events) {
            if (!Event.polarity()) continue;
            int Index = Event.y() * this->Width + Event.x();
            std::int64_t Prev = this->Last_Pos_Timestamp[Index];
            this->Last_Pos_Timestamp[Index] = Event.timestamp();
            if (Prev == 0) continue;

            std::int64_t dt_us = Event.timestamp() - Prev;
            if (dt_us > 100 && dt_us < 10000000) {
                double freq = 1e6 / static_cast<double>(dt_us);
                if (freq >= 50.0 && freq <= 5000.0)
                    Frequencies.push_back(freq);
            }
        }

        if (Frequencies.size() >= 5) {
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

    // ── Stage 2: Multi-car tracker (2 cars, 3 LEDs each = 6 LEDs total) ──
    Multi_Car_Tracker Tracker(2, 3, *resolution, 500000.0, 3.0);

    // ── Stage 3: Per-car frequency estimation ──
    std::vector<Frequency_Estimator> Car_Freq_Estimators;
    for (int i = 0; i < 2; ++i)
        Car_Freq_Estimators.emplace_back(*resolution);

    // ── Visualization ──
    dv::visualization::EventVisualizer Visualizer(*resolution,
        dv::visualization::colors::black,
        dv::visualization::colors::green,
        dv::visualization::colors::red);
    cv::namedWindow("Events", cv::WINDOW_NORMAL);
    cv::namedWindow("Tracking", cv::WINDOW_NORMAL);

    std::cerr << "Entering main loop — tracking 2 cars with 3 LEDs each..." << std::endl;

    while (Camera->isRunning()) {
        if (std::optional<dv::EventStore> Events = Camera->getNextEventBatch()) {
            if (!Events->isEmpty()) {
                // Stage 1
                Detector.Process_Batch(*Events);
                const std::vector<cv::Point2i>& Hot_Pixels = Detector.Get_Hot_Pixels();

                // Stage 2
                std::int64_t Timestamp = Events->getHighestTime();
                if (!Hot_Pixels.empty())
                    Tracker.Update(Hot_Pixels, Timestamp);

                // Stage 3: Per-car frequency
                double Car_Frequencies[2] = {0.0, 0.0};
                if (Tracker.Is_Initialized()) {
                    for (int c = 0; c < 2; ++c) {
                        dv::EventStore Car_Events = Tracker.Get_Events_For_Car(c, *Events);
                        Car_Frequencies[c] = Car_Freq_Estimators[c].Estimate(Car_Events);
                    }
                }

                // ── Draw ──
                cv::Mat Tracking_Frame(*resolution, CV_8UC3, cv::Vec3b(0, 0, 0));
                Detector.Highlight_Pixels(Tracking_Frame, cv::Vec3b(60, 60, 60));

                if (Tracker.Is_Initialized()) {
                    Tracker.Draw(Tracking_Frame);

                    // Display frequency per car
                    const auto& Cars = Tracker.Get_Cars();
                    cv::Scalar Colors[] = {cv::Scalar(0, 255, 0), cv::Scalar(0, 165, 255)};
                    for (int c = 0; c < 2; ++c) {
                        if (!Cars[c].Tracking) continue;

                        std::string Freq_Label;
                        if (Car_Frequencies[c] > 1.0) {
                            char buf[64];
                            std::snprintf(buf, sizeof(buf), "%.0f Hz", Car_Frequencies[c]);
                            Freq_Label = buf;
                        } else {
                            Freq_Label = "measuring...";
                        }

                        cv::putText(Tracking_Frame, Freq_Label,
                                    cv::Point(static_cast<int>(Cars[c].Centroid.x) - 25,
                                              static_cast<int>(Cars[c].Centroid.y) + 40),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.4, Colors[c], 1);
                    }
                } else {
                    cv::putText(Tracking_Frame, "Initializing...",
                                cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX,
                                0.5, cv::Scalar(0, 0, 255), 1);
                }

                cv::imshow("Events", Visualizer.generateImage(*Events));
                cv::imshow("Tracking", Tracking_Frame);
            }
        }
        cv::pollKey();
    }

    return 0;
}
