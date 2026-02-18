#include <cmath>
#include <chrono>
#include <deque>
#include <dv-processing/core/core.hpp>
#include <dv-processing/io/camera/discovery.hpp>            // Used for real-time readings
#include <dv-processing/io/mono_camera_recording.hpp>       // Used for reading .aedat4 recordings
#include <dv-processing/core/stream_slicer.hpp>             // Used to collect readings 
#include <dv-processing/visualization/event_visualizer.hpp> // Used to generate images to display
#include "PCA_Tracker.hpp"


PCA_Tracker::PCA_Tracker(size_t Max_Window_Size) : Max_Window_Size(Max_Window_Size) {}

void PCA_Tracker::Accept_Event_Batch(const dv::EventStore& Events) {
    for (const dv::Event& New_Event : Events) {
        // Push the new event into the rolling window
        this->Rolling_Window.push_back(New_Event);
        double New_X = New_Event.x();
        double New_Y = New_Event.y();

        this->Sum_X  += New_X;
        this->Sum_Y  += New_Y;
        this->Sum_XX += New_X * New_X;
        this->Sum_XY += New_X * New_Y;
        this->Sum_YY += New_Y * New_Y;
        
        // If the rolling window went over its defined size, remove the oldest event from the sum values
        if (this->Rolling_Window.size() > this->Max_Window_Size) {
            double Old_X = Rolling_Window.front().x();
            double Old_Y = Rolling_Window.front().y();
            this->Rolling_Window.pop_front();

            this->Sum_X  -= Old_X;
            this->Sum_Y  -= Old_Y;
            this->Sum_XX -= Old_X * Old_X;
            this->Sum_XY -= Old_X * Old_Y;
            this->Sum_YY -= Old_Y * Old_Y;
        }
    }
    
    // Calculate Mean and Covariance for the new state of the rolling window
    std::size_t Window_Size = this->Rolling_Window.size();
    this->Mean_X = this->Sum_X / Window_Size;
    this->Mean_Y = this->Sum_Y / Window_Size;
    // Check if there are at least two elements as Variance/Covariance requires at least 2 elements (N-1 degrees of freedom)
    if (Window_Size > 1) {
        this->Cov_XX = (this->Sum_XX - (this->Sum_X * this->Sum_X) / Window_Size) / (Window_Size - 1);
        this->Cov_XY = (this->Sum_XY - (this->Sum_X * this->Sum_Y) / Window_Size) / (Window_Size - 1);
        this->Cov_YY = (this->Sum_YY - (this->Sum_Y * this->Sum_Y) / Window_Size) / (Window_Size - 1);
    }

    // Call Calculate_PCA_Vectors() to update eigenvectors from new rolling window
    this->Calculate_PCA_Vectors();
}

void PCA_Tracker::Get_Means(double& Mean_X, double& Mean_Y) {
    // Write stored means into given return values
    Mean_X = this->Mean_X;
    Mean_Y = this->Mean_Y;
}

void PCA_Tracker::Get_Eigenvalues(double& Eigenvalue_1, double& Eigenvalue_2) {
    // Write stored eigenvalues into given return values
    Eigenvalue_1 = this->Eigenvalues[0];
    Eigenvalue_2 = this->Eigenvalues[1];
}

void PCA_Tracker::Get_Eigenvectors(double (*Eigenvectors)[2]) {
    // Write stored eigenvector components into the given 2D array
    Eigenvectors[0][0] = this->Eigenvectors[0][0];
    Eigenvectors[0][1] = this->Eigenvectors[0][1];
    Eigenvectors[1][0] = this->Eigenvectors[1][0];
    Eigenvectors[1][1] = this->Eigenvectors[1][1];
}

void PCA_Tracker::Draw_PCA_Vectors(cv::Mat& Frame, cv::Scalar Color_1, cv::Scalar Color_2, int Thickness) {
    // Create start and end points of the vector
    cv::Point Center_Point(static_cast<int>(this->Mean_X), static_cast<int>(this->Mean_Y));
    cv::Point Vector_End_1(
        static_cast<int>(this->Mean_X + this->Eigenvectors[0][0] * std::sqrt(this->Eigenvalues[0])),
        static_cast<int>(this->Mean_Y + this->Eigenvectors[0][1] * std::sqrt(this->Eigenvalues[0]))
    );
    cv::Point Vector_End_2(
        static_cast<int>(this->Mean_X + this->Eigenvectors[1][0] * std::sqrt(this->Eigenvalues[1])),
        static_cast<int>(this->Mean_Y + this->Eigenvectors[1][1] * std::sqrt(this->Eigenvalues[1]))
    );

    // Draw the two vectors onto the frame
    cv::arrowedLine(Frame, Center_Point, Vector_End_1, Color_1, Thickness);
    cv::arrowedLine(Frame, Center_Point, Vector_End_2, Color_2, Thickness);
}

void PCA_Tracker::Calculate_PCA_Vectors() {
    // Use quadratic formula to obtain eigenvalues (a = 1)
    double b = -(this->Cov_XX + this->Cov_YY);
    double c = ((this->Cov_XX * this->Cov_YY) - (this->Cov_XY * this->Cov_XY));
    double Discriminant = std::max(0.0, (b * b) - (4.0 * c)); // Used to prevent imaginary numbers and NAN return by std::sqrt
    this->Eigenvalues[0] = (-b + std::sqrt(Discriminant))/(2.0);
    this->Eigenvalues[1] = (-b - std::sqrt(Discriminant))/(2.0);
    
    // Create the first eigen vector and consider if the magnitude is 0
    double X_Component = this->Eigenvalues[0] - this->Cov_YY;
    double Y_Component = this->Cov_XY;
    double Magnitude = std::hypot(X_Component, Y_Component);
    if (Magnitude > 1e-9) { // Use a small epsilon instead of strictly 0
        this->Eigenvectors[0][0] = X_Component / Magnitude;
        this->Eigenvectors[0][1] = Y_Component / Magnitude;
    } else {
        // If Magnitude is 0, it means Cov_XY = 0 and Eigenvalue = Cov_YY.
        // This implies the eigenvector is simply the unit Y vector.
        this->Eigenvectors[0][0] = 0.0;
        this->Eigenvectors[0][1] = 1.0;
    }

    // Create the second eigen vector and consider if the magnitude is 0
    X_Component = this->Eigenvalues[1] - this->Cov_YY;
    Y_Component = this->Cov_XY;
    Magnitude = std::hypot(X_Component, Y_Component);
    if (Magnitude > 1e-9) { // Use a small epsilon instead of strictly 0
        this->Eigenvectors[1][0] = X_Component / Magnitude;
        this->Eigenvectors[1][1] = Y_Component / Magnitude;
    } else {
        // If Magnitude is 0, it means Cov_XY = 0 and Eigenvalue = Cov_YY.
        // This implies the eigenvector is simply the unit Y vector.
        this->Eigenvectors[1][0] = 0.0;
        this->Eigenvectors[1][1] = 1.0;
    }
}