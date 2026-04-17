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
#include "Frequency_Estimator.hpp"

// ─── Symbol Decoder ───────────────────────────────────────────────────────────
struct Symbol_Decoder {
    enum class State { Idle, Receiving };

    State       Current_State  = State::Idle;
    uint8_t     Dibit_Buffer   = 0;      // accumulates 4 dibits per character
    uint8_t     Dibit_Count    = 0;      // how many dibits received this char
    bool        Expect_Data    = false;  // true after a SEND, next symbol is data
    std::string Message        = "";     // fully decoded message so far

    // Call this every frame with the classified symbol for each LED
    // Returns true when a full message has been decoded (idle after data)
    bool Push_Symbol(const std::string& symbol) {
        if (symbol == "SEND") {
            Current_State = State::Receiving;
            Expect_Data   = true;
            return false;
        }

        if (symbol == "IDLE") {
            if (Current_State == State::Receiving && !Message.empty()) {
                // Flush any partial character
                Current_State = State::Idle;
                Dibit_Buffer  = 0;
                Dibit_Count   = 0;
                Expect_Data   = false;
                return true;  // message complete
            }
            return false;
        }

        // Data symbol — only valid after a SEND
        if (Current_State == State::Receiving && Expect_Data) {
            Expect_Data = false;

            uint8_t dibit = 0;
            if      (symbol == "00") dibit = 0;
            else if (symbol == "01") dibit = 1;
            else if (symbol == "10") dibit = 2;
            else if (symbol == "11") dibit = 3;
            else return false;  // unknown symbol, skip

            // LSB first — shift in from the top, we'll reverse at char boundary
            Dibit_Buffer |= (dibit << (Dibit_Count * 2));
            Dibit_Count++;

            if (Dibit_Count == 4) {
                // Full character received
                Message += static_cast<char>(Dibit_Buffer);
                Dibit_Buffer = 0;
                Dibit_Count  = 0;
            }
        }

        return false;
    }

    void Reset() {
        Current_State = State::Idle;
        Dibit_Buffer  = 0;
        Dibit_Count   = 0;
        Expect_Data   = false;
        Message       = "";
    }

    const std::string& Get_Message() const { return Message; }
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
    Kalman_LED_Tracker Tracker(*resolution, 3, 500000.0,  3.0,        3.0,    20,       120,      15,     8,    5,     0.0,   0.3);

    // ── Stage 3: Per-LED frequency estimation ──
    std::vector<Frequency_Estimator> Freq_Estimators;
    for (int i = 0; i < 3; ++i)
        Freq_Estimators.emplace_back(*resolution);
    std::vector<Symbol_Decoder> Decoders(3);

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

                        double Curr_Frequency = LED_Frequencies[i];
                        std::string symbol_name = "";
                        std::string Freq_Label  = "measuring...";

                        if (Curr_Frequency > 1.0) {
                            char buf[64];

                            if      (Curr_Frequency > 1100  && Curr_Frequency < 1300)  symbol_name = "IDLE";
                            else if (Curr_Frequency > 250  && Curr_Frequency < 350)  symbol_name = "00";
                            else if (Curr_Frequency > 350  && Curr_Frequency < 450)  symbol_name = "01";
                            else if (Curr_Frequency > 450  && Curr_Frequency < 550)  symbol_name = "10";
                            else if (Curr_Frequency > 550  && Curr_Frequency < 650) symbol_name = "11";
                            else if (Curr_Frequency > 800 && Curr_Frequency < 1000) symbol_name = "SEND";
                            else                                                       symbol_name = "ERROR";

                            // Feed into decoder
                            if (symbol_name != "ERROR") {
                                bool complete = Decoders[i].Push_Symbol(symbol_name);
                                if (complete) {
                                    std::cerr << "[LED " << i << "] Decoded: \""
                                            << Decoders[i].Get_Message() << "\"" << std::endl;
                                    Decoders[i].Reset();
                                }
                            }

                            std::snprintf(buf, sizeof(buf), "%s",
                                        //Curr_Frequency,
                                        //symbol_name.c_str(),
                                        Decoders[i].Get_Message().c_str());
                            Freq_Label = buf;
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
