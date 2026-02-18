#include <deque>
#include <dv-processing/core/core.hpp>


class PCA_Tracker {
    /* Private Data Members */
    private:
        // Configuration parameters
        std::deque<dv::Event> Rolling_Window;
        size_t Max_Window_Size;
        
        // The Rolling Sums (Internal math) (Should not get get functions)
        double Sum_X = 0, Sum_Y = 0; 
        double Sum_XX = 0, Sum_XY = 0, Sum_YY = 0;

        // The Results (Publicly readable through get functions)
        double Mean_X = 0, Mean_Y = 0;
        double Eigenvalues[2];
        double Eigenvectors[2][2];
        double Cov_XX = 0; double Cov_XY = 0; double Cov_YY = 0;

    /* Public functions */
    public:
        PCA_Tracker(size_t Max_Window_Size);
        void Accept_Event_Batch(const dv::EventStore& Events);
        void Get_Means(double& Mean_X, double& Mean_Y);
        void Get_Eigenvalues(double& Eigenvalue_1, double& Eigenvalue_2);
        void Get_Eigenvectors(double (*Eigenvectors)[2]);

    /* Private helper functions */
    private:
        void Calculate_PCA_Vectors();
};