#pragma once

#include <opencv2/core.hpp>

namespace msgf {

/**
 * @brief Single-image quality indicators (Mean, StdDev, IE, AG, PIQE).
 */
struct MsgfPlaneStats {
    bool valid = false;
    double mean = 0.0;
    double stdDev = 0.0;
    double ie = 0.0;     // Information Entropy (Eq. 20)
    double ag = 0.0;     // Average Gradient (Eq. 21)
    double piqe = 0.0;   // Perception-based Image Quality Evaluator (Venkatanath 2015)
};

/**
 * @brief Full quantitative comparison metrics between original and enhanced images.
 */
struct MsgfMetricResult {
    bool valid = false;
    double psnr = 0.0;   // Peak Signal-to-Noise Ratio in dB (Eq. 19)
    double ie   = 0.0;   // Output Information Entropy (Eq. 20)
    double ag   = 0.0;   // Output Average Gradient (Eq. 21)
    double piqe = 0.0;   // Output PIQE (lower is better)
    double mae  = 0.0;   // Mean Absolute Error [0, 255]
    MsgfPlaneStats inputStats;
    MsgfPlaneStats outputStats;
};

class MsgfMetrics {
public:
    static MsgfPlaneStats evaluateSingle(const cv::Mat& gray8U);
    static MsgfMetricResult evaluate(const cv::Mat& orig8U, const cv::Mat& enh8U);
};

} // namespace msgf

using MsgfPlaneStats = msgf::MsgfPlaneStats;
using MsgfMetricResult = msgf::MsgfMetricResult;
using MsgfMetrics = msgf::MsgfMetrics;
