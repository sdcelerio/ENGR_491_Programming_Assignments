#include <algorithm>
#include <cstdint>
#include <vector>
#include <dv-processing/core/core.hpp>
#include <opencv2/core.hpp>

#include "Frequency_Estimator.hpp"

Frequency_Estimator::Frequency_Estimator(cv::Size Resolution)
    : Width(Resolution.width),
      Last_Pos_Timestamp(Resolution.width * Resolution.height, 0) {}

double Frequency_Estimator::Estimate(const dv::EventStore& Events) {
    // If input Events are empty abort
    if (Events.isEmpty()) 
        return this->Measured_Frequency;

    // Create temporary vector storing the measured frequencies
    std::vector<double> Frequencies;
    Frequencies.reserve(Events.size() / 2);

    // Operate through the given events
    for (const dv::Event& Event : Events) {
        // If the event is the LED turning off, ignore it
        if (!Event.polarity())
            continue;

        int Index = Event.y() * this->Width + Event.x();
        std::int64_t Prev = this->Last_Pos_Timestamp[Index];
        this->Last_Pos_Timestamp[Index] = Event.timestamp();

        // Ignore duplicate events if that can happen
        if (Prev == 0) 
            continue;

        // Guard against zero/negative dt (timestamp jitter or duplicate events)
        std::int64_t dt_us = Event.timestamp() - Prev;
        if (dt_us <= 0)
            continue;

        // Clamp to valid frequency range: 100Hz–2000Hz
        // 100Hz → dt = 10000µs, 2000Hz → dt = 500µs
        if (dt_us < 500 || dt_us > 10000)
            continue;

        double freq = 1e6 / static_cast<double>(dt_us);
        Frequencies.push_back(freq);
    }

    // If there are enough estimated frequencies get the median
    if (Frequencies.size() >= 5) {
        std::sort(Frequencies.begin(), Frequencies.end());
        this-> Measured_Frequency = Frequencies[Frequencies.size() / 2];
    }

    return this->Measured_Frequency;
}

double Frequency_Estimator::Get_Frequency() const {
    return this->Measured_Frequency;
}

void Frequency_Estimator::Reset() {
    // Reset all timestamps to 0 and measured frequency to 0
    std::fill(this->Last_Pos_Timestamp.begin(), this->Last_Pos_Timestamp.end(), 0);
    this->Measured_Frequency = 0.0;
}