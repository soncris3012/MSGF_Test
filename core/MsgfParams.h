#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace msgf {

/**
 * @brief Tunable parameters for the 4-stage MSGF-TIR pipeline
 *        (Li et al., Fire 2024, 7, 192).
 */
struct MsgfParams {
    // 1. 3-scale Self-Guided Filter (Section 2.1, Eq. 7-9)
    // r: local window radius (kernel size = 2*r + 1)
    // eps: regularization parameter on normalized intensity domain [0, 1]
    int r1 = 2;       double eps1 = 0.005;   // Small-scale detail D1
    int r2 = 6;       double eps2 = 0.02;    // Medium-scale detail D2
    int r3 = 16;      double eps3 = 0.08;    // Large-scale detail D3

    // 2. Base layer CLAHE enhancement (Section 2.2, Eq. 10-15)
    double claheClipLimit = 2.0;             // Contrast clipping factor c
    int claheTileGrid = 8;                   // Grid n x n (e.g. 8x8 tiles)

    // 3. Dynamic Linear Enhancement for D1, D2, D3 (Section 2.3, Eq. 16-17)
    double alpha1 = 1.5;
    double alpha2 = 1.5;
    double alpha3 = 1.5;

    // 4. Detail combination weights (Figure 1) & Fusion coefficients (Section 2.4, Eq. 18)
    double w1 = 1.0;
    double w2 = 1.0;
    double w3 = 1.0;

    // I_out = p * I_B + q * I_D
    double p = 1.0;
    double q = 1.0;
};

/**
 * @brief Intermediate and final outputs produced by MsgfAlgorithm::run().
 *        Detail layers D1..D3, D1_enh..D3_enh, and ID_fused are kept as signed
 *        CV_32FC1 matrices to avoid clipping dark edges prior to linear fusion.
 */
struct MsgfOutputs {
    cv::Mat input8U;               // CV_8UC1 [0, 255]
    cv::Mat B1, B2, B3;            // Base layers at scales 1..3, CV_32FC1 [0, 1]
    cv::Mat D1, D2, D3;            // Detail layers at scales 1..3, CV_32FC1 (signed)
    cv::Mat D1_enh, D2_enh, D3_enh;// Enhanced detail layers, CV_32FC1 (signed)
    cv::Mat IB_clahe;              // CLAHE-enhanced base layer, CV_32FC1 [0, 1]
    cv::Mat ID_fused;              // Weighted sum of enhanced details, CV_32FC1 (signed)
    cv::Mat output8U;              // Clamped final output, CV_8UC1 [0, 255]
    double elapsedMs = 0.0;        // Processing latency (ms)
};

/**
 * @brief Reference row from Table 1 of Li et al. (Fire 2024, 7, 192).
 */
struct PaperBenchmarkRow {
    std::string sceneName;
    std::string datasetSource;
    std::string algorithm;
    double psnr = 0.0;
    double ie = 0.0;
    double ag = 0.0;
    double piqe = 0.0;
    bool isProposed = false;
};

inline const std::vector<PaperBenchmarkRow>& paperTable1Benchmarks()
{
    static const std::vector<PaperBenchmarkRow> rows = {
        {"Scene 1", "iRay T3-Pro", "HE",            13.75, 6.95, 86.20, 38.79, false},
        {"Scene 1", "iRay T3-Pro", "GF&DDE",        23.42, 6.93, 48.72, 38.91, false},
        {"Scene 1", "iRay T3-Pro", "BF&DDE",        27.07, 7.05, 49.22, 66.21, false},
        {"Scene 1", "iRay T3-Pro", "MSGF-TIR (Ours)", 27.07, 7.34, 54.15, 34.45, true},

        {"Scene 2", "FLIR",        "HE",            12.33, 5.91, 61.83, 48.54, false},
        {"Scene 2", "FLIR",        "GF&DDE",        22.83, 6.09, 18.43, 11.32, false},
        {"Scene 2", "FLIR",        "BF&DDE",        19.56, 6.11, 19.48, 42.97, false},
        {"Scene 2", "FLIR",        "MSGF-TIR (Ours)", 21.20, 6.47, 19.84,  9.35, true},

        {"Scene 3", "FLIR",        "HE",            11.80, 5.66, 86.92, 46.18, false},
        {"Scene 3", "FLIR",        "GF&DDE",        18.57, 5.70, 15.88,  9.38, false},
        {"Scene 3", "FLIR",        "BF&DDE",        21.46, 5.79, 15.79, 47.56, false},
        {"Scene 3", "FLIR",        "MSGF-TIR (Ours)", 30.74, 6.16, 19.05,  8.02, true},
    };
    return rows;
}

} // namespace msgf

using MsgfParams = msgf::MsgfParams;
using MsgfOutputs = msgf::MsgfOutputs;
using PaperBenchmarkRow = msgf::PaperBenchmarkRow;
