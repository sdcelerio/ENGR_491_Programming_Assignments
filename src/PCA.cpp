#include <iostream>
#include <deque>
#include <filesystem>
#include <dv-processing/core/core.hpp>
#include <dv-processing/io/camera/discovery.hpp>            // Used for real-time readings
#include <dv-processing/io/mono_camera_recording.hpp>       // Used for reading .aedat4 recordings
#include <dv-processing/visualization/event_visualizer.hpp> // Used to generate images to display
#include <dv-processing/core/stream_slicer.hpp>             // Used to collect readings 
#include <opencv4/opencv2/highgui.hpp>                      // Used to display the data

#define ROLLING_WINDOW_SIZE 20000
#define SLICER_SIZE 5000

void Calculate_PCA(const double Cov_XX, const double Cov_XY, const double Cov_YY, double Eigenvalues[], double Eignevector_1[], double Eigenvector_2[]);
void Draw_Vector(cv::Mat& Frame, double Center_X, double Center_Y, double Vector[], double Magnitude, cv::Scalar Color);

int main(void) {
    // Open the first camera that appears
    dv::io::camera::CameraPtr Camera = dv::io::camera::open();
    
    /*
    // Initialize the reader for the file
    std::filesystem::path filePath = "/mnt/c/Users/sdcel/OneDrive/Documents/School_Work/College_Work/4th_Year/2026_Spring/ENGR_491_512/Camera_Assignments/Ruler.aedat4";
    dv::io::MonoCameraRecording Reader(filePath);
    if (!Reader.isEventStreamAvailable()) {
        std::cerr << "Error! Could not find any events in the filepath " << filePath << std::endl;
        return 1;
    }   
    */
    
    // Intialize the Visualizer
    dv::visualization::EventVisualizer visualizer(Camera->getEventResolution().value(), dv::visualization::colors::black,
        dv::visualization::colors::green, dv::visualization::colors::red);
    
    // Intialize the Event Slicer
    dv::EventStreamSlicer Slicer;
    std::deque<dv::Event> Rolling_Window;
    double Mean_X = 0; double Mean_Y = 0;
    double Sum_X = 0; double Sum_Y = 0; double Sum_XX = 0; double Sum_XY = 0; double Sum_YY = 0;
    double Cov_XX = 0; double Cov_XY = 0; double Cov_YY = 0;

    std::chrono::microseconds time(100000);
    Slicer.doEveryTimeInterval(time, [&](const dv::EventStore &Events) {
        for (const dv::Event& New_Event : Events) {
            // Push the new event into the rolling window
            Rolling_Window.push_back(New_Event);
            double New_X = New_Event.x();
            double New_Y = New_Event.y();
            Sum_X  += New_X;
            Sum_Y  += New_Y;
            Sum_XX += New_X * New_X;
            Sum_YY += New_Y * New_Y;
            Sum_XY += New_X * New_Y;

            // If the rolling window went over its defined size, remove the oldest event from the sum values
            if (Rolling_Window.size() > ROLLING_WINDOW_SIZE) {
                double Old_X = Rolling_Window.front().x();
                double Old_Y = Rolling_Window.front().y();
                Rolling_Window.pop_front();

                Sum_X  -= Old_X;
                Sum_Y  -= Old_Y;
                Sum_XX -= Old_X * Old_X;
                Sum_XY -= Old_X * Old_Y;
                Sum_YY -= Old_Y * Old_Y;
            }
            
            // Calculate Mean and Covariance on the fly for the current state of the rolling window
            std::size_t Window_Size = Rolling_Window.size();
            
            Mean_X = Sum_X / Window_Size;
            Mean_Y = Sum_Y / Window_Size;

            // Variance/Covariance requires at least 2 elements (N-1 degrees of freedom)
            if (Window_Size > 1) {
                Cov_XX = (Sum_XX - (Sum_X * Sum_X) / Window_Size) / (Window_Size - 1);
                Cov_YY = (Sum_YY - (Sum_Y * Sum_Y) / Window_Size) / (Window_Size - 1);
                Cov_XY = (Sum_XY - (Sum_X * Sum_Y) / Window_Size) / (Window_Size - 1);
            }
        }

        // Calculate the PCA values 
        double Eigenvalues[2];
        double Vector_1[2];
        double Vector_2[2];
        Calculate_PCA(Cov_XX, Cov_XY, Cov_YY, Eigenvalues, Vector_1, Vector_2);
        
        // Produce frame and output
        cv::Mat frame = visualizer.generateImage(Events);
        cv::Point center(static_cast<int>(Mean_X), static_cast<int>(Mean_Y)); // Draw a solid blue circle at the mean
        Draw_Vector(frame, Mean_X, Mean_Y, Vector_1, std::sqrt(Eigenvalues[0]), cv::Scalar(255, 255, 255));
        Draw_Vector(frame, Mean_X, Mean_Y, Vector_2, std::sqrt(Eigenvalues[1]), cv::Scalar(255, 255, 255));
        cv::circle(frame, center, 5, cv::Scalar(255, 0, 0), -1);
        cv::imshow("Real-Time 500 Events", frame);
        cv::pollKey();
    });
    
    // Loop to feed event readings into the slicer 
    while(Camera->isRunning()) {
       if (std::optional<dv::EventStore> Events = Camera->getNextEventBatch()) 
            Slicer.accept(*Events);
    }

    return 0;
}

void Calculate_PCA(double Cov_XX, double Cov_XY, double Cov_YY, double Eigenvalues[], double Eignevector_1[], double Eigenvector_2[]) {
    // Use quadratic formula to obtain eigenvalues (a = 1)
    double b = -(Cov_XX + Cov_YY);
    double c = ((Cov_XX * Cov_YY) - (Cov_XY * Cov_XY));
    double Discriminant = std::max(0.0, (b * b) - (4. * c)); // Used to prevent imaginary numbers and NAN return by std::sqrt
    Eigenvalues[0] = (-b + std::sqrt(Discriminant))/(2.0);
    Eigenvalues[1] = (-b - std::sqrt(Discriminant))/(2.0);
    
    // Create vectors and return 
    double Magnitude = std::sqrt(std::pow(Eigenvalues[0] - Cov_YY, 2) + std::pow(Cov_XY, 2));
    Eignevector_1[0] = (Eigenvalues[0] - Cov_YY)/Magnitude; 
    Eignevector_1[1] = Cov_XY/Magnitude;
    if (Magnitude == 0) {
        std::cout << "This is the problem" << std::endl;
    }
    
    Magnitude = std::sqrt(std::pow(Eigenvalues[1] - Cov_YY, 2) + std::pow(Cov_XY, 2));
    Eigenvector_2[0] = (Eigenvalues[1] - Cov_YY)/Magnitude; 
    Eigenvector_2[1] = Cov_XY/Magnitude;
    if (Magnitude == 0) {
        std::cout << "This is the problem" << std::endl;
    }
}

void Draw_Vector(cv::Mat& Frame, double Center_X, double Center_Y, double Vector[], double Magnitude, cv::Scalar Color) {
    cv::Point start(static_cast<int>(Center_X), static_cast<int>(Center_Y));
    cv::Point end(
        static_cast<int>(Center_X + Vector[0] * Magnitude),
        static_cast<int>(Center_Y + Vector[1] * Magnitude)
    );

    // 2. Draw the arrow
    // tipLength is the fractional length of the arrow head (0.3 = 30% of line length)
    cv::arrowedLine(Frame, start, end, Color, 2, cv::LINE_AA, 0, 0.3);
}