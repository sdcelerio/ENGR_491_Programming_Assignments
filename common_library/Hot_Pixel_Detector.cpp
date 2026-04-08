#include <vector>
#include <cstdint>
#include <cstring>
#include <dv-processing/core/core.hpp>
#include <opencv2/core.hpp>
#include "Hot_Pixel_Detector.hpp"

Hot_Pixel_Detector::Hot_Pixel_Detector(cv::Size Resolution, int Threshold)
    : Resolution(Resolution),
      Threshold(Threshold),
      Counts(Resolution.width * Resolution.height, 0) {

    this->Hot_Pixels.reserve(Resolution.area() / 10);
}

void Hot_Pixel_Detector::Process_Batch(const dv::EventStore& Events) {
    // Reset counts and hot pixels
    std::memset(this->Counts.data(), 0, this->Counts.size() * sizeof(std::int32_t));
    this->Hot_Pixels.clear();

    if (Events.isEmpty())
        return;

    // Count events per pixel
    for (const dv::Event& Event : Events) {
        this->Counts[Event.y() * this->Resolution.width + Event.x()]++;
    }

    // Collect pixels that exceed the threshold
    for (int y = 0; y < this->Resolution.height; ++y) {
        const std::int32_t* Row = this->Counts.data() + y * this->Resolution.width;
        for (int x = 0; x < this->Resolution.width; ++x) {
            if (Row[x] >= this->Threshold)
                this->Hot_Pixels.emplace_back(x, y);
        }
    }
}

const std::vector<cv::Point2i>& Hot_Pixel_Detector::Get_Hot_Pixels() const {
    return this->Hot_Pixels;
}

void Hot_Pixel_Detector::Highlight_Pixels(cv::Mat& Frame, cv::Vec3b Color) const {
    for (const cv::Point2i& Pixel : this->Hot_Pixels)
        Frame.at<cv::Vec3b>(Pixel.y, Pixel.x) = Color;
}
