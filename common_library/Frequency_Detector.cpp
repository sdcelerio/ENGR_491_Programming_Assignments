#include <vector>
#include <cmath>
#include <cstdint>
#include <dv-processing/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "Frequency_Detector.hpp"

Frequency_Detector::Frequency_Detector(int16_t Width, int16_t Height, double Target_Frequency, double Tolerance, int Required_Matches)
    : Width(Width), Height(Height), Target_Frequency(Target_Frequency), Tolerance(Tolerance), Required_Matches(Required_Matches) {

    // Initialize state arrays
    this->Pixel_States.resize(Width * Height);
}

void Frequency_Detector::Accept_Event_Batch(const dv::EventStore& Events) {
    for (const dv::Event& Event : Events) {
        // We only measure positive polarity (OFF to ON transitions) to capture full cycles
        if (!Event.polarity())
            continue;

        // Ignore impossibly short displacements to filter out hardware noise bursts
        int Index = Event.y() * this->Width + Event.x();
        int64_t Time_Displacement_us = Event.timestamp() - this->Pixel_States[Index].Latest_Timestamp; // Time period in microseconds
        this->Pixel_States[Index].Latest_Timestamp = Event.timestamp();
        if (Time_Displacement_us < 1000) 
            continue;

        // Check if the measured frequeny is the desired target within the tolerance. Update state variables
        double Measured_Frequency = 1e6f / static_cast<float>(Time_Displacement_us);
        if (std::abs(Measured_Frequency - this->Target_Frequency) <= this->Tolerance) 
            this->Pixel_States[Index].Num_Matches++;
        else 
            this->Pixel_States[Index].Num_Matches = 0; // Reset streak on mismatch
    }
}

void Frequency_Detector::Highlight_Pixels(cv::Mat& Frame, cv::Vec3b Color) {
    for (int Index = 0; Index < this->Width * this->Height; Index++) {
        if (this->Pixel_States[Index].Num_Matches >= this->Required_Matches) {
            int x = Index % this->Width, y = Index / this->Width;
            Frame.at<cv::Vec3b>(y, x) = Color;
        }   
    } 
}