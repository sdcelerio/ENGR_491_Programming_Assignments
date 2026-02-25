#include <vector>
#include <cmath>
#include <cstdint>
#include <dv-processing/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "Frequency_Detector.hpp"

Frequency_Detector::Frequency_Detector(cv::Size Size, double Target_Frequency, double Tolerance, int Required_Matches)
    : Size(Size), Target_Frequency(Target_Frequency), Tolerance(Tolerance), Required_Matches(Required_Matches) {

    // Initialize state arrays
    this->Pixel_States.resize(Size.area());
    this->Valid_Indexes.reserve((Size.area()) / 10); // Reserves space so reallocation is minimum at the start
}

void Frequency_Detector::Accept_Event_Batch(const dv::EventStore& Events) {
    // Check if the event store passed is empty to avoid buggy behavior
    if (Events.isEmpty())
        return;

    // Remove valid pixels that haven't fired recently enough to still be valid
    std::int64_t Latest_Timestamp = Events.getHighestTime();
    this->Remove_Old_Pixels(Latest_Timestamp);

    // Check every new event and increment if it has occurred recently
    for (const dv::Event& Event : Events) {
        // We only measure positive polarity (OFF to ON transitions) to capture full cycles
        if (!Event.polarity())
            continue;
        
        // Ignore impossibly short displacements to filter out hardware noise bursts
        int Index = Event.y() * this->Size.width + Event.x();
        std::int64_t Time_Displacement_us = Event.timestamp() - this->Pixel_States[Index].Latest_Timestamp; // Time period in microseconds
        this->Pixel_States[Index].Latest_Timestamp = Event.timestamp();
        if (Time_Displacement_us < 1000) 
            continue;

        // Check if the measured frequency is the desired target within the tolerance. Update state variables
        double Measured_Frequency = 1e6 / static_cast<double>(Time_Displacement_us);
        if (std::abs(Measured_Frequency - this->Target_Frequency) <= this->Tolerance) {
            this->Pixel_States[Index].Num_Matches++;

            // If the pixel reaches the required matches, consider it valid and save its index into Valid_Indexes 
            if (this->Pixel_States[Index].Num_Matches >= this->Required_Matches) {
                this->Pixel_States[Index].Num_Matches = this->Required_Matches; // Prevents integer overflows

                // Ensures that there are no duplicate indexes
                if (!this->Pixel_States[Index].Is_Valid) {
                    this->Pixel_States[Index].Is_Valid = true;
                    this->Valid_Indexes.push_back(Index);
                }
            }
        } 
        // Else the detected frequency is not within the target so reset
        else {
            this->Pixel_States[Index].Num_Matches = 0;
            this->Pixel_States[Index].Is_Valid = false;
        }
    }
}

void Frequency_Detector::Highlight_Pixels(cv::Mat& Frame, cv::Vec3b Color) const {
    for (std::uint32_t Index : this->Valid_Indexes) {
        int x = Index % this->Size.width, y = Index / this->Size.width;
        Frame.at<cv::Vec3b>(y, x) = Color;
    }
}

void Frequency_Detector::Remove_Old_Pixels(std::int64_t Latest_Timestamp) {
    // Check each valid pixel and check if its too old. Perform pop and swap for efficient erasing
    std::int64_t Expiry_Threshold = static_cast<int64_t>(4e6 / (this->Target_Frequency)); // Effective time period twice of the target frequency period
    std::size_t Valid_Index = 0;
    while(Valid_Index < this->Valid_Indexes.size()) {
        std::uint32_t Pixel_Index = Valid_Indexes[Valid_Index];
        if (Latest_Timestamp - this->Pixel_States[Pixel_Index].Latest_Timestamp > Expiry_Threshold) {
            this->Pixel_States[Pixel_Index] = PixelState();
            this->Valid_Indexes[Valid_Index] = this->Valid_Indexes.back();
            this->Valid_Indexes.pop_back();
        } 
        else 
            Valid_Index++;
    }
}