#pragma once

#include "core/MsgfParams.h"

namespace msgf {

/**
 * @brief Pure C++17 / OpenCV implementation of MSGF-TIR:
 *        "Thermal Infrared-Image-Enhancement Algorithm Based on Multi-Scale Guided Filtering"
 *        (Li et al., Fire 2024, 7, 192).
 */
class MsgfAlgorithm {
public:
    /**
     * @brief Execute the complete 4-stage MSGF-TIR enhancement pipeline.
     * @param inputGrayOrBgr 2D input image (CV_8U, CV_16U, CV_32F/64F; 1/3/4 channels).
     * @param params Algorithm parameters.
     * @return All intermediate layers (B1..B3, D1..D3, IB_clahe, ID_fused) and output8U.
     */
    static MsgfOutputs run(const cv::Mat& inputGrayOrBgr, const MsgfParams& params);

    /**
     * @brief Normalize a signed float layer (e.g., D1, D2, D3, ID_fused) to [0, 255] CV_8UC1
     *        for visualization, matching Figure 3(b-d) in the paper.
     */
    static cv::Mat normalizeForDisplay(const cv::Mat& layer32F);

    /**
     * @brief Convert arbitrary supported image formats into a clean 8-bit grayscale matrix.
     */
    static cv::Mat toGray8U(const cv::Mat& inputGrayOrBgr);

    /**
     * @brief Self-Guided Filter following Eq. (7)-(9).
     * @param I Single-channel CV_32FC1 input & guidance image.
     * @param radius Window radius r (kernel width = 2*r + 1).
     * @param eps Regularization parameter epsilon.
     */
    static cv::Mat selfGuidedFilter(const cv::Mat& I, int radius, double eps);

    /**
     * @brief Dynamic Linear Enhancement for a signed detail layer following Eq. (16)-(17).
     */
    static cv::Mat dynamicLinearEnhance(const cv::Mat& detailLayer, double alpha);
};

} // namespace msgf

using MsgfAlgorithm = msgf::MsgfAlgorithm;
