#include <vector>
#include <cmath>
#include <cstdint>
#include <dv-processing/core/core.hpp>
#include <opencv2/core.hpp>
#include "Frequency_Detector.hpp"

Frequency_Detector::Frequency_Detector(const cv::Size Resolution, double Target_Frequency, double Tolerance, int Required_Matches)
    : Resolution(Resolution), Target_Frequency(Target_Frequency), Tolerance(Tolerance), Required_Matches(Required_Matches) {

    // Initialize state arrays
    this->Expiry_Threshold = static_cast<int64_t>(3e6 / (this->Target_Frequency)); // Effective time period twice of the target frequency period
    this->Pixel_States.resize(Resolution.area());
    this->Valid_Pixels.reserve((Resolution.area()) / 10); // Reserves space so reallocation is minimum at the start
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
        
        // Based on the time period between the previous timestamp and new timestamp, calculate the measured frequency
        int Index = Event.y() * this->Resolution.width + Event.x();
        std::int64_t Time_Displacement_us = Event.timestamp() - this->Pixel_States[Index].Latest_Timestamp; // Time period in microseconds
        this->Pixel_States[Index].Latest_Timestamp = Event.timestamp();
        if (Time_Displacement_us < 1000) // Ignore impossibly short displacements to filter out hardware noise bursts
            continue;

        // Check if the measured frequency is the desired target within the tolerance. Update state variables
        double Measured_Frequency = 1e6 / static_cast<double>(Time_Displacement_us);
        if (std::abs(Measured_Frequency - this->Target_Frequency) <= this->Tolerance) {
            // If the pixel reaches the required matches, consider it valid and save its index into Valid_Pixels 
            this->Pixel_States[Index].Num_Matches++;
            if (this->Pixel_States[Index].Num_Matches >= this->Required_Matches) {
                this->Pixel_States[Index].Num_Matches = this->Required_Matches; // Prevents integer overflows

                // Check to see if the pixel is already considered valid. Ensures that there are no duplicate indexes
                if (this->Pixel_States[Index].Index_In_Valid == -1) {
                    this->Pixel_States[Index].Index_In_Valid = (std::int32_t) this->Valid_Pixels.size();
                    this->Valid_Pixels.emplace_back(Event.x(), Event.y());
                }
            }
        } 
        // Else the detected frequency is not within the target so reset
        else {
            this->Pixel_States[Index].Num_Matches = 0;
            if (this->Pixel_States[Index].Index_In_Valid != -1) {   // If the pixel was considered valid, perform a pop and swap for O(1) vector deletion
                this->Swap_Pop(this->Pixel_States[Index].Index_In_Valid);
                this->Pixel_States[Index].Index_In_Valid = -1;
            }
        }
    }
}

dv::EventStore Frequency_Detector::Generate_Events() {
    dv::EventStore Return_Store;
    for (cv::Point2i Valid_Pixel : this->Valid_Pixels)
        Return_Store.emplace_back(0, Valid_Pixel.x, Valid_Pixel.y, true);
    
    return Return_Store;
}

const std::vector<cv::Point2i>& Frequency_Detector::Get_Valid_Pixels() {
    return this->Valid_Pixels;
}

void Frequency_Detector::Highlight_Pixels(cv::Mat& Frame, cv::Vec3b Color) const {
    // Draw all the valid pixels onto the frame given the color
    for (cv::Point2i Valid_Pixel : this->Valid_Pixels)
        Frame.at<cv::Vec3b>(Valid_Pixel.y, Valid_Pixel.x) = Color;
}

void Frequency_Detector::Remove_Old_Pixels(std::int64_t Latest_Events_Timestamp) {
    // Check each valid pixel and check if its too old by utilizing the Expiry_Threshold. Perform pop and swap for O(1) vector deletion
    std::size_t Valid_Index = 0;
    while(Valid_Index < this->Valid_Pixels.size()) {
        std::uint32_t Pixel_Index = this->Valid_Pixels[Valid_Index].y * this->Resolution.width + this->Valid_Pixels[Valid_Index].x;
        if (Latest_Events_Timestamp - this->Pixel_States[Pixel_Index].Latest_Timestamp > this->Expiry_Threshold) {
            this->Pixel_States[Pixel_Index] = PixelState();
            this->Swap_Pop(Valid_Index);
        } 
        else 
            Valid_Index++;
    }
}

void Frequency_Detector::Swap_Pop(std::int32_t Target_Index) {
    // If the target index to remove from valid vector is not the back, proceed to swap. This prevents segmentation faults
    if (Target_Index != (std::int32_t) this->Valid_Pixels.size() - 1) {
        std::uint32_t Pixel_Index = this->Valid_Pixels.back().y * this->Resolution.width + this->Valid_Pixels.back().x;
        this->Pixel_States[Pixel_Index].Index_In_Valid = Target_Index;
        this->Valid_Pixels[Target_Index] = this->Valid_Pixels.back();
    }

    this->Valid_Pixels.pop_back();
}