#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <dv-processing/core/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "Kalman_LED_Tracker.hpp"


/* ──────────────── Constructor ──────────────── */
Kalman_LED_Tracker::Kalman_LED_Tracker(cv::Size Resolution,
                                       int      Target_Cluster_Count,
                                       double   Process_Noise_Sigma,
                                       double   Measurement_Noise_Sigma,
                                       double   Gate_Sigma,
                                       int      Min_Gate_Half_Size,
                                       int      Max_Gate_Half_Size,
                                       int      Min_Hot_Pixels,
                                       int      Lost_Threshold,
                                       int      Lost_Grace_Frames,
                                       double   Expected_Ratio,
                                       double   Ratio_Tolerance)
    : Resolution(Resolution),
      Target_Cluster_Count(Target_Cluster_Count),
      Process_Noise_Sigma(Process_Noise_Sigma),
      Measurement_Noise_Sigma(Measurement_Noise_Sigma),
      Gate_Sigma(Gate_Sigma),
      Min_Gate_Half_Size(Min_Gate_Half_Size),
      Max_Gate_Half_Size(Max_Gate_Half_Size),
      Min_Hot_Pixels(Min_Hot_Pixels),
      Lost_Threshold(Lost_Threshold),
      Lost_Grace_Frames(Lost_Grace_Frames),
      Expected_Ratio(Expected_Ratio),
      Ratio_Tolerance(Ratio_Tolerance) {

    this->LEDs.resize(Target_Cluster_Count);

    // Measurement matrix H (2x6): observe position only
    this->H = cv::Mat::zeros(2, 6, CV_64F);
    this->H.at<double>(0, 0) = 1.0;
    this->H.at<double>(1, 1) = 1.0;

    // Measurement noise R (2x2)
    this->R = cv::Mat::zeros(2, 2, CV_64F);
    this->R.at<double>(0, 0) = Measurement_Noise_Sigma * Measurement_Noise_Sigma;
    this->R.at<double>(1, 1) = Measurement_Noise_Sigma * Measurement_Noise_Sigma;
}


/* ──────────────── Public Functions ──────────────── */
void Kalman_LED_Tracker::Update(const std::vector<cv::Point2i>& Hot_Pixels, std::int64_t Timestamp) {
    if (!this->Initialized) {
        this->Initialized = this->Initialize(Hot_Pixels);
        if (this->Initialized)
            this->Last_Timestamp = Timestamp;
        return;
    }

    double dt = static_cast<double>(Timestamp - this->Last_Timestamp) / 1e6;
    this->Last_Timestamp = Timestamp;
    if (dt <= 0.0 || dt > 1.0) return;

    this->Track(Hot_Pixels, dt);
}

dv::EventStore Kalman_LED_Tracker::Get_Events_In_Gate(int LED_Index, const dv::EventStore& Events) const {
    dv::EventStore Filtered;
    if (LED_Index < 0 || LED_Index >= static_cast<int>(this->LEDs.size())) return Filtered;
    if (this->LEDs[LED_Index].State_Flag == Status::Lost) return Filtered;

    cv::Rect Gate = this->Get_Gate_Rect(LED_Index);
    for (const dv::Event& Event : Events) {
        if (Gate.contains(cv::Point2i(Event.x(), Event.y())))
            Filtered.emplace_back(Event.timestamp(), Event.x(), Event.y(), Event.polarity());
    }
    return Filtered;
}

const std::vector<Kalman_LED_Tracker::TrackedLED>& Kalman_LED_Tracker::Get_LEDs() const { return this->LEDs; }
bool Kalman_LED_Tracker::Is_Initialized() const { return this->Initialized; }

void Kalman_LED_Tracker::Reset() {
    this->Initialized = false;
    for (TrackedLED& LED : this->LEDs) {
        LED.State_Flag = Status::Lost; LED.Event_Count = 0; LED.Lost_Frames = 0;
    }
}

void Kalman_LED_Tracker::Draw(cv::Mat& Frame, cv::Scalar Gate_Color, cv::Scalar Velocity_Color) const {
    for (int i = 0; i < static_cast<int>(this->LEDs.size()); ++i) {
        const TrackedLED& LED = this->LEDs[i];
        if (LED.State_Flag == Status::Lost) continue;

        double px = LED.State.at<double>(0), py = LED.State.at<double>(1);
        double vx = LED.State.at<double>(2), vy = LED.State.at<double>(3);
        cv::Point Center(static_cast<int>(px), static_cast<int>(py));

        cv::Rect Gate = this->Get_Gate_Rect(i);
        cv::rectangle(Frame, Gate, Gate_Color, 1);
        cv::circle(Frame, Center, 3, Gate_Color, -1);

        cv::Point Vel_End(static_cast<int>(px + vx * 0.3), static_cast<int>(py + vy * 0.3));
        cv::arrowedLine(Frame, Center, Vel_End, Velocity_Color, 1);

        std::string Label = "LED " + std::to_string(i) + " [" + std::to_string(LED.Event_Count) + "]";
        cv::putText(Frame, Label, cv::Point(Gate.x, Gate.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, Gate_Color, 1);
    }
}


/* ──────────────── Initialization ──────────────── */
bool Kalman_LED_Tracker::Initialize(const std::vector<cv::Point2i>& Hot_Pixels) {
    if (static_cast<int>(Hot_Pixels.size()) < this->Min_Hot_Pixels) return false;

    cv::Mat Points(static_cast<int>(Hot_Pixels.size()), 2, CV_32F);
    for (int i = 0; i < static_cast<int>(Hot_Pixels.size()); ++i) {
        Points.at<float>(i, 0) = static_cast<float>(Hot_Pixels[i].x);
        Points.at<float>(i, 1) = static_cast<float>(Hot_Pixels[i].y);
    }

    cv::Mat Labels, Centers;
    cv::kmeans(Points, this->Target_Cluster_Count, Labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 1.0),
               5, cv::KMEANS_PP_CENTERS, Centers);

    PCA_Result PCA = this->Compute_PCA(Centers);

    // ── Per-cluster covariance validation ──
    // Compute covariance (trace) of each cluster. Reject if clusters are too dissimilar
    // (indicates k-means split one LED or merged two)
    int N = this->Target_Cluster_Count;
    std::vector<double> Cluster_Traces(N, 0.0);
    std::vector<double> Cluster_Elongations(N, 0.0);
    std::vector<int> Cluster_Counts_Init(N, 0);

    for (int c = 0; c < N; ++c) {
        // Gather points for this cluster
        double sum_x = 0, sum_y = 0;
        int count = 0;
        for (int p = 0; p < Labels.rows; ++p) {
            if (Labels.at<int>(p) == c) {
                sum_x += Hot_Pixels[p].x;
                sum_y += Hot_Pixels[p].y;
                count++;
            }
        }
        Cluster_Counts_Init[c] = count;
        if (count < 3) {
            std::cerr << "  INIT REJECTED: cluster " << c << " has only " << count << " points" << std::endl;
            return false;
        }

        double mean_x = sum_x / count, mean_y = sum_y / count;

        // Compute 2x2 covariance for this cluster
        double cxx = 0, cxy = 0, cyy = 0;
        for (int p = 0; p < Labels.rows; ++p) {
            if (Labels.at<int>(p) == c) {
                double dx = Hot_Pixels[p].x - mean_x;
                double dy = Hot_Pixels[p].y - mean_y;
                cxx += dx * dx; cxy += dx * dy; cyy += dy * dy;
            }
        }
        cxx /= (count - 1); cxy /= (count - 1); cyy /= (count - 1);

        // Trace = total spread of this cluster
        Cluster_Traces[c] = cxx + cyy;

        // Elongation = eigenvalue ratio within this cluster
        double b = -(cxx + cyy), det = cxx * cyy - cxy * cxy;
        double disc = std::max(0.0, b * b - 4.0 * det);
        double lam1 = (-b + std::sqrt(disc)) / 2.0;
        double lam2 = (-b - std::sqrt(disc)) / 2.0;
        Cluster_Elongations[c] = (lam2 > 1e-6) ? lam1 / lam2 : 1e6;

        std::cerr << "  Cluster " << c << ": count=" << count 
                  << " trace=" << Cluster_Traces[c]
                  << " elongation=" << Cluster_Elongations[c] << std::endl;
    }

    // Check 1: Reject if any cluster is extremely elongated (likely a split blob)
    for (int c = 0; c < N; ++c) {
        if (Cluster_Elongations[c] > 10.0) {
            std::cerr << "  INIT REJECTED: cluster " << c << " too elongated (" << Cluster_Elongations[c] << ")" << std::endl;
            return false;
        }
    }

    // Check 2: Reject if cluster sizes are too different (one is >5x larger than another)
    double max_trace = *std::max_element(Cluster_Traces.begin(), Cluster_Traces.end());
    double min_trace = *std::min_element(Cluster_Traces.begin(), Cluster_Traces.end());
    if (min_trace > 0 && max_trace / min_trace > 5.0) {
        std::cerr << "  INIT REJECTED: cluster sizes too different (ratio=" << max_trace / min_trace << ")" << std::endl;
        return false;
    }

    // Check 3: Reject if cluster point counts are too imbalanced (one has >5x more points)
    int max_count = *std::max_element(Cluster_Counts_Init.begin(), Cluster_Counts_Init.end());
    int min_count = *std::min_element(Cluster_Counts_Init.begin(), Cluster_Counts_Init.end());
    if (min_count > 0 && max_count / min_count > 5) {
        std::cerr << "  INIT REJECTED: point counts too imbalanced (" << max_count << " vs " << min_count << ")" << std::endl;
        return false;
    }

    // ── Arrangement validation (3+ LEDs) ──
    if (this->Target_Cluster_Count >= 3) {
        std::cerr << "  Arrangement: ratio=" << PCA.Ratio << std::endl;
        if (this->Expected_Ratio > 0.0) {
            double err = std::abs(PCA.Ratio - this->Expected_Ratio) / this->Expected_Ratio;
            if (err > this->Ratio_Tolerance) {
                std::cerr << "  INIT REJECTED: arrangement ratio=" << PCA.Ratio << " expected=" << this->Expected_Ratio << std::endl;
                return false;
            }
        }
    } else {
        // For 2 LEDs: simple separation check
        float dx = Centers.at<float>(0, 0) - Centers.at<float>(1, 0);
        float dy = Centers.at<float>(0, 1) - Centers.at<float>(1, 1);
        if (std::sqrt(dx * dx + dy * dy) < 20.0) {
            std::cerr << "  INIT REJECTED: centers too close" << std::endl;
            return false;
        }
    }

    // Identity assignment
    std::vector<int> Identity_Map;
    if (this->Has_Reference) {
        Identity_Map = this->Match_Identities(PCA);
    } else {
        std::vector<int> Sort_Order(this->Target_Cluster_Count);
        std::iota(Sort_Order.begin(), Sort_Order.end(), 0);
        std::sort(Sort_Order.begin(), Sort_Order.end(), [&](int a, int b) {
            return PCA.Local_Coords[a].x < PCA.Local_Coords[b].x;
        });
        Identity_Map.resize(this->Target_Cluster_Count);
        for (int led = 0; led < this->Target_Cluster_Count; ++led)
            Identity_Map[Sort_Order[led]] = led;
        this->Reference_Local_Coords.resize(this->Target_Cluster_Count);
        for (int k = 0; k < this->Target_Cluster_Count; ++k)
            this->Reference_Local_Coords[Identity_Map[k]] = PCA.Local_Coords[k];
        this->Reference_Ratio = PCA.Ratio;
        this->Has_Reference = true;
    }

    std::cerr << "INIT SUCCESS: " << Hot_Pixels.size() << " hot pixels" << std::endl;
    for (int k = 0; k < this->Target_Cluster_Count; ++k) {
        int led = Identity_Map[k];
        double cx = Centers.at<float>(k, 0), cy = Centers.at<float>(k, 1);

        // 6-state: [px, py, vx, vy, ax, ay]
        this->LEDs[led].State = cv::Mat::zeros(6, 1, CV_64F);
        this->LEDs[led].State.at<double>(0) = cx;
        this->LEDs[led].State.at<double>(1) = cy;

        this->LEDs[led].Covariance = cv::Mat::zeros(6, 6, CV_64F);
        this->LEDs[led].Covariance.at<double>(0, 0) = 400.0;     // px
        this->LEDs[led].Covariance.at<double>(1, 1) = 400.0;     // py
        this->LEDs[led].Covariance.at<double>(2, 2) = 10000.0;   // vx
        this->LEDs[led].Covariance.at<double>(3, 3) = 10000.0;   // vy
        this->LEDs[led].Covariance.at<double>(4, 4) = 50000.0;   // ax
        this->LEDs[led].Covariance.at<double>(5, 5) = 50000.0;   // ay

        this->LEDs[led].Local_PCA_Coords = PCA.Local_Coords[k];
        this->LEDs[led].Event_Count = Cluster_Counts_Init[k];
        this->LEDs[led].Lost_Frames = 0;
        this->LEDs[led].State_Flag = Status::Tracking;

        std::cerr << "  LED " << led << " center=(" << cx << "," << cy << ")" << std::endl;
    }
    return true;
}


/* ──────────────── Tracking ──────────────── */
void Kalman_LED_Tracker::Track(const std::vector<cv::Point2i>& Hot_Pixels, double dt) {
    cv::Mat F, Q;
    this->Build_Prediction_Matrices(dt, F, Q);
    int N = this->Target_Cluster_Count;

    // ── Step 1: Predict all LEDs ──
    for (TrackedLED& LED : this->LEDs) {
        LED.State = F * LED.State;
        LED.Covariance = F * LED.Covariance * F.t() + Q;

        // Clamp position to frame
        LED.State.at<double>(0) = std::clamp(LED.State.at<double>(0), 0.0, static_cast<double>(this->Resolution.width - 1));
        LED.State.at<double>(1) = std::clamp(LED.State.at<double>(1), 0.0, static_cast<double>(this->Resolution.height - 1));

        // Covariance ceiling on position — prevents gate explosion
        LED.Covariance.at<double>(0, 0) = std::min(LED.Covariance.at<double>(0, 0), 10000.0);
        LED.Covariance.at<double>(1, 1) = std::min(LED.Covariance.at<double>(1, 1), 10000.0);
    }

    // ── Step 2: Exclusive pixel assignment ──
    std::vector<std::vector<cv::Point2i>> Pixels_Per_LED(N);
    for (int i = 0; i < N; ++i) Pixels_Per_LED[i].reserve(256);

    for (const cv::Point2i& Pixel : Hot_Pixels) {
        double Best_Dist_Sq = std::numeric_limits<double>::max();
        int Best_LED = -1;

        for (int i = 0; i < N; ++i) {
            cv::Rect Gate = this->Get_Gate_Rect(i);
            if (!Gate.contains(Pixel)) continue;

            double dx = Pixel.x - this->LEDs[i].State.at<double>(0);
            double dy = Pixel.y - this->LEDs[i].State.at<double>(1);
            double d = dx * dx + dy * dy;
            if (d < Best_Dist_Sq) { Best_Dist_Sq = d; Best_LED = i; }
        }

        if (Best_LED >= 0)
            Pixels_Per_LED[Best_LED].push_back(Pixel);
    }

    // ── Step 3: Update each LED ──
    int Lost_Count = 0;
    cv::Mat I = cv::Mat::eye(6, 6, CV_64F);

    for (int i = 0; i < N; ++i) {
        TrackedLED& LED = this->LEDs[i];
        LED.Event_Count = static_cast<int>(Pixels_Per_LED[i].size());

        if (LED.Event_Count < this->Lost_Threshold) {
            LED.Lost_Frames++;
            // Decay velocity and acceleration when coasting
            LED.State.at<double>(2) *= 0.85;  // vx
            LED.State.at<double>(3) *= 0.85;  // vy
            LED.State.at<double>(4) *= 0.5;   // ax
            LED.State.at<double>(5) *= 0.5;   // ay
            if (LED.Lost_Frames > this->Lost_Grace_Frames) {
                LED.State_Flag = Status::Lost;
                Lost_Count++;
            }
            continue;
        }

        LED.Lost_Frames = 0;
        LED.State_Flag = Status::Tracking;

        // Measure
        cv::Point2d Centroid = Compute_Centroid(Pixels_Per_LED[i]);
        cv::Mat z = (cv::Mat_<double>(2, 1) << Centroid.x, Centroid.y);

        // Kalman update (Joseph form)
        cv::Mat y = z - this->H * LED.State;
        cv::Mat S = this->H * LED.Covariance * this->H.t() + this->R;
        cv::Mat K = LED.Covariance * this->H.t() * S.inv();
        LED.State = LED.State + K * y;
        cv::Mat IKH = I - K * this->H;
        LED.Covariance = IKH * LED.Covariance * IKH.t() + K * this->R * K.t();

        // Covariance floor on position — prevents gate collapse
        LED.Covariance.at<double>(0, 0) = std::max(LED.Covariance.at<double>(0, 0), 25.0);
        LED.Covariance.at<double>(1, 1) = std::max(LED.Covariance.at<double>(1, 1), 25.0);
    }

    if (Lost_Count > 0) {
        std::cerr << "TRACKING RESET: " << Lost_Count << " LEDs lost" << std::endl;
        this->Reset();
    }
}


/* ──────────────── Prediction Matrices (Constant Acceleration) ──────────────── */
void Kalman_LED_Tracker::Build_Prediction_Matrices(double dt, cv::Mat& F, cv::Mat& Q) const {
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double dt4 = dt3 * dt;
    double dt5 = dt4 * dt;
    double s2 = this->Process_Noise_Sigma * this->Process_Noise_Sigma;

    // State transition: [px, py, vx, vy, ax, ay]
    F = cv::Mat::eye(6, 6, CV_64F);
    F.at<double>(0, 2) = dt;           // px += vx*dt
    F.at<double>(0, 4) = 0.5 * dt2;    // px += 0.5*ax*dt²
    F.at<double>(1, 3) = dt;           // py += vy*dt
    F.at<double>(1, 5) = 0.5 * dt2;    // py += 0.5*ay*dt²
    F.at<double>(2, 4) = dt;           // vx += ax*dt
    F.at<double>(3, 5) = dt;           // vy += ay*dt

    // Process noise: jerk model
    // Per-axis noise vector G = [dt²/2, dt, 1]ᵀ maps jerk to [pos, vel, accel]
    Q = cv::Mat::zeros(6, 6, CV_64F);
    // X axis (0, 2, 4)
    Q.at<double>(0, 0) = s2 * dt5 / 20.0;
    Q.at<double>(0, 2) = s2 * dt4 / 8.0;
    Q.at<double>(0, 4) = s2 * dt3 / 6.0;
    Q.at<double>(2, 0) = s2 * dt4 / 8.0;
    Q.at<double>(2, 2) = s2 * dt3 / 3.0;
    Q.at<double>(2, 4) = s2 * dt2 / 2.0;
    Q.at<double>(4, 0) = s2 * dt3 / 6.0;
    Q.at<double>(4, 2) = s2 * dt2 / 2.0;
    Q.at<double>(4, 4) = s2 * dt;
    // Y axis (1, 3, 5)
    Q.at<double>(1, 1) = s2 * dt5 / 20.0;
    Q.at<double>(1, 3) = s2 * dt4 / 8.0;
    Q.at<double>(1, 5) = s2 * dt3 / 6.0;
    Q.at<double>(3, 1) = s2 * dt4 / 8.0;
    Q.at<double>(3, 3) = s2 * dt3 / 3.0;
    Q.at<double>(3, 5) = s2 * dt2 / 2.0;
    Q.at<double>(5, 1) = s2 * dt3 / 6.0;
    Q.at<double>(5, 3) = s2 * dt2 / 2.0;
    Q.at<double>(5, 5) = s2 * dt;
}


/* ──────────────── Gate Rectangle ──────────────── */
cv::Rect Kalman_LED_Tracker::Get_Gate_Rect(int LED_Index) const {
    const TrackedLED& LED = this->LEDs[LED_Index];
    double px = LED.State.at<double>(0), py = LED.State.at<double>(1);
    double Pxx = LED.Covariance.at<double>(0, 0), Pyy = LED.Covariance.at<double>(1, 1);

    double half_w = std::clamp(this->Gate_Sigma * std::sqrt(std::max(Pxx, 0.0)),
                               static_cast<double>(this->Min_Gate_Half_Size),
                               static_cast<double>(this->Max_Gate_Half_Size));
    double half_h = std::clamp(this->Gate_Sigma * std::sqrt(std::max(Pyy, 0.0)),
                               static_cast<double>(this->Min_Gate_Half_Size),
                               static_cast<double>(this->Max_Gate_Half_Size));

    cv::Rect Gate(static_cast<int>(px - half_w), static_cast<int>(py - half_h),
                  static_cast<int>(2.0 * half_w), static_cast<int>(2.0 * half_h));
    Gate &= cv::Rect(0, 0, this->Resolution.width, this->Resolution.height);
    return Gate;
}


/* ──────────────── PCA ──────────────── */
Kalman_LED_Tracker::PCA_Result Kalman_LED_Tracker::Compute_PCA(const cv::Mat& Centers) const {
    PCA_Result R;
    int N = Centers.rows;
    R.Mean = {0, 0};
    for (int i = 0; i < N; ++i) { R.Mean.x += Centers.at<float>(i, 0); R.Mean.y += Centers.at<float>(i, 1); }
    R.Mean.x /= N; R.Mean.y /= N;

    double Cxx = 0, Cxy = 0, Cyy = 0;
    for (int i = 0; i < N; ++i) {
        double dx = Centers.at<float>(i, 0) - R.Mean.x, dy = Centers.at<float>(i, 1) - R.Mean.y;
        Cxx += dx*dx; Cxy += dx*dy; Cyy += dy*dy;
    }
    if (N > 1) { Cxx /= (N-1); Cxy /= (N-1); Cyy /= (N-1); }

    double b = -(Cxx + Cyy), c = Cxx*Cyy - Cxy*Cxy;
    double D = std::max(0.0, b*b - 4*c);
    R.Eigenvalues[0] = (-b + std::sqrt(D)) / 2.0;
    R.Eigenvalues[1] = (-b - std::sqrt(D)) / 2.0;

    for (int i = 0; i < 2; ++i) {
        double ex = R.Eigenvalues[i] - Cyy, ey = Cxy, mag = std::hypot(ex, ey);
        if (mag > 1e-9) { R.Eigenvectors[i][0] = ex/mag; R.Eigenvectors[i][1] = ey/mag; }
        else { R.Eigenvectors[i][0] = (i==0)?1:0; R.Eigenvectors[i][1] = (i==0)?0:1; }
    }

    R.Ratio = (R.Eigenvalues[1] > 1e-6) ? R.Eigenvalues[0]/R.Eigenvalues[1] : 1e6;

    R.Local_Coords.resize(N);
    for (int i = 0; i < N; ++i) {
        double dx = Centers.at<float>(i, 0) - R.Mean.x, dy = Centers.at<float>(i, 1) - R.Mean.y;
        R.Local_Coords[i].x = dx*R.Eigenvectors[0][0] + dy*R.Eigenvectors[0][1];
        R.Local_Coords[i].y = dx*R.Eigenvectors[1][0] + dy*R.Eigenvectors[1][1];
    }
    return R;
}


/* ──────────────── Identity Matching ──────────────── */
std::vector<int> Kalman_LED_Tracker::Match_Identities(const PCA_Result& PCA) const {
    int N = this->Target_Cluster_Count;
    std::vector<int> Best_Map(N); double Best_Cost = 1e18;

    for (int flip = 0; flip < 4; ++flip) {
        double s1 = (flip&1)?-1:1, s2 = (flip&2)?-1:1;
        std::vector<cv::Point2d> Adj(N);
        for (int i = 0; i < N; ++i) { Adj[i].x = s1*PCA.Local_Coords[i].x; Adj[i].y = s2*PCA.Local_Coords[i].y; }

        std::vector<bool> Used(N, false); std::vector<int> Map(N, -1); double Cost = 0;
        for (int ref = 0; ref < N; ++ref) {
            double MinD = 1e18; int BK = -1;
            for (int k = 0; k < N; ++k) {
                if (Used[k]) continue;
                double dx = Adj[k].x - this->Reference_Local_Coords[ref].x;
                double dy = Adj[k].y - this->Reference_Local_Coords[ref].y;
                double d = dx*dx + dy*dy;
                if (d < MinD) { MinD = d; BK = k; }
            }
            Map[BK] = ref; Used[BK] = true; Cost += MinD;
        }
        if (Cost < Best_Cost) { Best_Cost = Cost; Best_Map = Map; }
    }
    return Best_Map;
}


/* ──────────────── Centroid ──────────────── */
cv::Point2d Kalman_LED_Tracker::Compute_Centroid(const std::vector<cv::Point2i>& Points) {
    double Sx = 0, Sy = 0;
    for (const cv::Point2i& P : Points) { Sx += P.x; Sy += P.y; }
    double N = static_cast<double>(Points.size());
    return {Sx/N, Sy/N};
}
