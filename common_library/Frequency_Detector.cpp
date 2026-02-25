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
        
        // Based on the time period between the previous timestamp and 
        int Index = Event.y() * this->Size.width + Event.x();
        std::int64_t Time_Displacement_us = Event.timestamp() - this->Pixel_States[Index].Latest_Timestamp; // Time period in microseconds
        this->Pixel_States[Index].Latest_Timestamp = Event.timestamp();
        if (Time_Displacement_us < 1000) // Ignore impossibly short displacements to filter out hardware noise bursts
            continue;

        // Check if the measured frequency is the desired target within the tolerance. Update state variables
        double Measured_Frequency = 1e6 / static_cast<double>(Time_Displacement_us);
        if (std::abs(Measured_Frequency - this->Target_Frequency) <= this->Tolerance) {
            // If the pixel reaches the required matches, consider it valid and save its index into Valid_Indexes 
            this->Pixel_States[Index].Num_Matches++;
            if (this->Pixel_States[Index].Num_Matches >= this->Required_Matches) {
                this->Pixel_States[Index].Num_Matches = this->Required_Matches; // Prevents integer overflows

                // Check to see if the pixel is already considered valid. Ensures that there are no duplicate indexes
                if (this->Pixel_States[Index].Valid_Index == -1) {
                    this->Pixel_States[Index].Valid_Index = (std::int32_t) this->Valid_Indexes.size();
                    this->Valid_Indexes.push_back(Index);
                }
            }
        } 
        // Else the detected frequency is not within the target so reset
        else {
            this->Pixel_States[Index].Num_Matches = 0;
            if (this->Pixel_States[Index].Valid_Index != -1) {   // If the pixel was considered valid, perform a pop and swap for O(1) vector deletion
                std::int32_t Remove_Index = this->Pixel_States[Index].Valid_Index;
                if (Remove_Index != (std::int32_t) this->Valid_Indexes.size() - 1) {
                    this->Pixel_States[this->Valid_Indexes.back()].Valid_Index = Remove_Index;
                    this->Valid_Indexes[Remove_Index] = this->Valid_Indexes.back();
                }
                this->Valid_Indexes.pop_back();
                this->Pixel_States[Index].Valid_Index = -1;
            }
        }
    }
}

void Frequency_Detector::Highlight_Pixels(cv::Mat& Frame, cv::Vec3b Color) const {
    for (std::uint32_t Index : this->Valid_Indexes) {
        int x = Index % this->Size.width, y = Index / this->Size.width;
        Frame.at<cv::Vec3b>(y, x) = Color;
    }
}

void Frequency_Detector::Remove_Old_Pixels(std::int64_t Latest_Events_Timestamp) {
    // Check each valid pixel and check if its too old. Perform pop and swap for efficient erasing
    std::int64_t Expiry_Threshold = static_cast<int64_t>(4e6 / (this->Target_Frequency)); // Effective time period twice of the target frequency period
    std::size_t Valid_Index = 0;
    while(Valid_Index < this->Valid_Indexes.size()) {
        std::uint32_t Pixel_Index = Valid_Indexes[Valid_Index];
        if (Latest_Events_Timestamp - this->Pixel_States[Pixel_Index].Latest_Timestamp > Expiry_Threshold) {
            this->Pixel_States[Pixel_Index] = PixelState();
            if (Valid_Index != this->Valid_Indexes.size() - 1) {
                this->Pixel_States[this->Valid_Indexes.back()].Valid_Index = (std::int32_t) Valid_Index;
                this->Valid_Indexes[Valid_Index] = this->Valid_Indexes.back();
            }
            this->Valid_Indexes.pop_back();
        } 
        else 
            Valid_Index++;
    }
}