#include "core/MsgfBaselines.h"

#include "core/MsgfAlgorithm.h"

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <stdexcept>

namespace msgf {

std::string MsgfBaselines::algorithmName(BenchmarkAlgorithmId id)
{
    switch (id) {
    case BenchmarkAlgorithmId::Original: return "Original";
    case BenchmarkAlgorithmId::HE:       return "HE";
    case BenchmarkAlgorithmId::GF_DDE:   return "GF&DDE";
    case BenchmarkAlgorithmId::BF_DDE:   return "BF&DDE";
    case BenchmarkAlgorithmId::MSGF_TIR: return "MSGF-TIR";
    case BenchmarkAlgorithmId::N5:       return "N5";
    }
    return "Unknown";
}

BaselineRunResult MsgfBaselines::runHE(const cv::Mat& inputGrayOrBgr)
{
    const auto t0 = std::chrono::steady_clock::now();
    const cv::Mat in8U = MsgfAlgorithm::toGray8U(inputGrayOrBgr);
    cv::Mat out8U;
    cv::equalizeHist(in8U, out8U);
    const auto t1 = std::chrono::steady_clock::now();
    return {
        std::move(out8U),
        std::chrono::duration<double, std::milli>(t1 - t0).count()
    };
}

BaselineRunResult MsgfBaselines::runGFDDE(const cv::Mat& inputGrayOrBgr)
{
    const auto t0 = std::chrono::steady_clock::now();
    const cv::Mat in8U = MsgfAlgorithm::toGray8U(inputGrayOrBgr);

    cv::Mat I;
    in8U.convertTo(I, CV_32F, 1.0 / 255.0);

    // Single-scale Guided Filter decomposition (radius = 4, eps = 0.02)
    const cv::Mat base = MsgfAlgorithm::selfGuidedFilter(I, 4, 0.02);
    const cv::Mat detail = I - base;

    // Contrast-limited dynamic range compression on base + linear detail boost
    cv::Mat baseClamped;
    cv::min(cv::max(base, 0.0f), 1.0f, baseClamped);
    cv::Mat base8U;
    baseClamped.convertTo(base8U, CV_8U, 255.0);

    auto clahe = cv::createCLAHE(1.5, cv::Size(
        std::max(1, std::min(8, base8U.cols)),
        std::max(1, std::min(8, base8U.rows))
    ));
    cv::Mat baseEnh8U;
    clahe->apply(base8U, baseEnh8U);

    cv::Mat baseEnh;
    baseEnh8U.convertTo(baseEnh, CV_32F, 1.0 / 255.0);

    const cv::Mat detailEnh = MsgfAlgorithm::dynamicLinearEnhance(detail, 1.35);
    cv::Mat fused = baseEnh + detailEnh;
    cv::min(cv::max(fused, 0.0f), 1.0f, fused);

    cv::Mat out8U;
    fused.convertTo(out8U, CV_8U, 255.0);

    const auto t1 = std::chrono::steady_clock::now();
    return {
        std::move(out8U),
        std::chrono::duration<double, std::milli>(t1 - t0).count()
    };
}

BaselineRunResult MsgfBaselines::runBFDDE(const cv::Mat& inputGrayOrBgr)
{
    const auto t0 = std::chrono::steady_clock::now();
    const cv::Mat in8U = MsgfAlgorithm::toGray8U(inputGrayOrBgr);

    cv::Mat I;
    in8U.convertTo(I, CV_32F, 1.0 / 255.0);

    // Bilateral filtering decomposition (spatial sigma = 4.0, range sigma = 0.12)
    cv::Mat base;
    cv::bilateralFilter(I, base, 9, 0.12, 4.0, cv::BORDER_REFLECT);
    const cv::Mat detail = I - base;

    cv::Mat baseClamped;
    cv::min(cv::max(base, 0.0f), 1.0f, baseClamped);
    cv::Mat base8U;
    baseClamped.convertTo(base8U, CV_8U, 255.0);

    auto clahe = cv::createCLAHE(1.5, cv::Size(
        std::max(1, std::min(8, base8U.cols)),
        std::max(1, std::min(8, base8U.rows))
    ));
    cv::Mat baseEnh8U;
    clahe->apply(base8U, baseEnh8U);

    cv::Mat baseEnh;
    baseEnh8U.convertTo(baseEnh, CV_32F, 1.0 / 255.0);

    const cv::Mat detailEnh = MsgfAlgorithm::dynamicLinearEnhance(detail, 1.35);
    cv::Mat fused = baseEnh + detailEnh;
    cv::min(cv::max(fused, 0.0f), 1.0f, fused);

    cv::Mat out8U;
    fused.convertTo(out8U, CV_8U, 255.0);

    const auto t1 = std::chrono::steady_clock::now();
    return {
        std::move(out8U),
        std::chrono::duration<double, std::milli>(t1 - t0).count()
    };
}

BaselineRunResult MsgfBaselines::runAlgorithm(
    BenchmarkAlgorithmId id,
    const cv::Mat& inputGrayOrBgr,
    const MsgfParams& msgfParams,
    const N5Params& n5Params)
{
    switch (id) {
    case BenchmarkAlgorithmId::Original: {
        cv::Mat orig = MsgfAlgorithm::toGray8U(inputGrayOrBgr);
        return {std::move(orig), 0.0};
    }
    case BenchmarkAlgorithmId::HE:
        return runHE(inputGrayOrBgr);
    case BenchmarkAlgorithmId::GF_DDE:
        return runGFDDE(inputGrayOrBgr);
    case BenchmarkAlgorithmId::BF_DDE:
        return runBFDDE(inputGrayOrBgr);
    case BenchmarkAlgorithmId::MSGF_TIR: {
        const MsgfOutputs out = MsgfAlgorithm::run(inputGrayOrBgr, msgfParams);
        return {out.output8U, out.elapsedMs};
    }
    case BenchmarkAlgorithmId::N5: {
        const N5Outputs out = N5Algorithm::run(inputGrayOrBgr, n5Params);
        return {out.output8U, out.elapsedMs};
    }
    }
    throw std::invalid_argument("Unsupported BenchmarkAlgorithmId");
}

} // namespace msgf
