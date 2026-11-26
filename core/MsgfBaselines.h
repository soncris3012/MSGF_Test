#pragma once

#include "core/MsgfMetrics.h"
#include "core/MsgfParams.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace msgf {

enum class BenchmarkAlgorithmId {
    Original = 0,
    HE,
    GF_DDE,
    BF_DDE,
    MSGF_TIR
};

struct BaselineRunResult {
    cv::Mat output8U;
    double elapsedMs = 0.0;
};

/**
 * @brief Classical baseline infrared enhancement algorithms from Table 1
 *        of Li et al. (Fire 2024, 7, 192) for live dataset benchmarking.
 */
class MsgfBaselines {
public:
    static std::string algorithmName(BenchmarkAlgorithmId id);

    /**
     * @brief Global Histogram Equalization (HE) via cv::equalizeHist.
     */
    static BaselineRunResult runHE(const cv::Mat& inputGrayOrBgr);

    /**
     * @brief Single-scale Guided Filter + Dynamic Detail Enhancement (GF&DDE).
     */
    static BaselineRunResult runGFDDE(const cv::Mat& inputGrayOrBgr);

    /**
     * @brief Bilateral Filter + Dynamic Detail Enhancement (BF&DDE).
     */
    static BaselineRunResult runBFDDE(const cv::Mat& inputGrayOrBgr);

    /**
     * @brief Dispatch any supported algorithm uniformly.
     */
    static BaselineRunResult runAlgorithm(
        BenchmarkAlgorithmId id,
        const cv::Mat& inputGrayOrBgr,
        const MsgfParams& msgfParams
    );
};

} // namespace msgf
