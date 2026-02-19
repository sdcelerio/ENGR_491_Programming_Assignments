#include <deque>
#include <dv-processing/core/core.hpp>
#include <opencv4/opencv2/highgui.hpp>

class PCA_Tracker {
    /* Private Data Members */
    private:
        // Configuration parameters
        std::size_t Max_Window_Size;
        
        // The Rolling Sums (Internal math) (Should not have get functions)
        std::deque<dv::Event> Rolling_Window;
        double Sum_X = 0, Sum_Y = 0; 
        double Sum_XX = 0, Sum_XY = 0, Sum_YY = 0;
        double Cov_XX = 0; double Cov_XY = 0; double Cov_YY = 0;

        // The Results (Publicly readable through get functions)
        double Mean_X = 0, Mean_Y = 0;
        double Eigenvalues[2];
        double Eigenvectors[2][2];

    /* Public functions */
    public:
        PCA_Tracker(size_t Max_Window_Size);
        void Accept_Event_Batch(const dv::EventStore& Events);
        void Get_Means(double& Mean_X, double& Mean_Y);
        void Get_Eigenvalues(double& Eigenvalue_1, double& Eigenvalue_2);
        void Get_Eigenvectors(double (*Eigenvectors)[2]);
        void Draw_PCA_Vectors(cv::Mat& Frame, cv::Scalar Color_1, cv::Scalar Color_2, int Thickness);

    /* Private helper functions */
    private:
        void Calculate_PCA_Vectors();
};