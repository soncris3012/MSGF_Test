#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace msgf {

/**
 * @brief Parameters for the N5 Thermal Enhancer (Accepted N5 baseline defaults).
 *        Default values reproduce the frozen N5 snapshot bit-for-bit.
 */
struct N5Params {
    int stage1Window = 7;          // Guided base width w1 (must be odd >= 3 or >= 1)
    double claheClipLimit = 2.0;   // CLAHE clip limit c
    int claheTileGrid = 8;         // CLAHE grid n x n
    int stage2Window = 3;          // Fine guided base width w2
    double postHalfAttenuation = 0.5; // Post-Half attenuation factor on fine residual R2
};

/**
 * @brief Complete intermediate and final outputs from N5Algorithm::run().
 */
struct N5Outputs {
    cv::Mat input8U;               // Prepared 2D CV_8UC1 plane P
    cv::Mat B1;                    // Stage-1 guided base (CV_64FC1)
    cv::Mat R1;                    // Stage-1 signed residual x - B1 (CV_64FC1)
    cv::Mat C_clahe8U;             // CLAHE-mapped base C (CV_8UC1)
    cv::Mat U_stage18U;            // Stage-1 quantized intermediate U (CV_8UC1)
    cv::Mat B2;                    // Stage-2 fine guided base (CV_64FC1)
    cv::Mat R2;                    // Stage-2 signed residual u - B2 (CV_64FC1)
    cv::Mat output8U;              // Final Post-Half output Y (CV_8UC1)
    double baseRegularization = 0.0;
    double smallRegularization = 0.0;
    bool constantPreserved = false;
    double elapsedMs = 0.0;
};

/**
 * @brief Bit-exact implementation of Accepted N5 (Guided-CLAHE + Post-Half).
 *        Also supports parametric ablation via N5Params.
 */
class N5Algorithm {
public:
    static N5Outputs run(const cv::Mat& inputGrayOrBgr, const N5Params& params = N5Params{});
    static cv::Mat normalizeForDisplay(const cv::Mat& layer64F);
};

} // namespace msgf

// Exact upstream API compatibility from Accepted N5 snapshot
namespace qart::core::enhancement {

struct N5EnhancementResult {
    cv::Mat outputImage;
    double smallRegularization = 0.0;
    bool constantPreserved = false;
};

[[nodiscard]] cv::Mat prepareThermalUint8Plane(const cv::Mat& image);
[[nodiscard]] cv::Mat bypassThermal(const cv::Mat& plane);

/**
 * @brief Frozen Accepted N5 processor (strict CV_8UC1 input contract).
 *        Give each worker thread its own instance (CLAHE object is stateful).
 */
class N5ThermalEnhancer final {
public:
    N5ThermalEnhancer();
    N5ThermalEnhancer(const N5ThermalEnhancer&) = delete;
    N5ThermalEnhancer& operator=(const N5ThermalEnhancer&) = delete;
    N5ThermalEnhancer(N5ThermalEnhancer&&) = delete;
    N5ThermalEnhancer& operator=(N5ThermalEnhancer&&) = delete;

    [[nodiscard]] N5EnhancementResult enhance(const cv::Mat& plane);

private:
    cv::Ptr<cv::CLAHE> clahe_;
};

} // namespace qart::core::enhancement
