#include "core/MsgfAlgorithm.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace msgf {

cv::Mat MsgfAlgorithm::toGray8U(const cv::Mat& inputGrayOrBgr)
{
    if (inputGrayOrBgr.empty() || inputGrayOrBgr.dims != 2) {
        throw std::invalid_argument("MsgfAlgorithm requires a non-empty 2D image");
    }

    cv::Mat singleChannel;
    const int ch = inputGrayOrBgr.channels();
    if (ch == 1) {
        singleChannel = inputGrayOrBgr;
    } else if (ch == 3) {
        cv::cvtColor(inputGrayOrBgr, singleChannel, cv::COLOR_BGR2GRAY);
    } else if (ch == 4) {
        cv::cvtColor(inputGrayOrBgr, singleChannel, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::invalid_argument("Unsupported channel count for thermal enhancement");
    }

    cv::Mat out8U;
    const int depth = singleChannel.depth();
    if (depth == CV_8U) {
        out8U = singleChannel.clone();
    } else if (depth == CV_16U) {
        cv::normalize(singleChannel, out8U, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);
    } else if (depth == CV_32F || depth == CV_64F) {
        double minVal = 0.0;
        double maxVal = 0.0;
        cv::minMaxLoc(singleChannel, &minVal, &maxVal);
        if (minVal >= 0.0 && maxVal <= 1.0) {
            singleChannel.convertTo(out8U, CV_8U, 255.0);
        } else {
            cv::normalize(singleChannel, out8U, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);
        }
    } else {
        cv::normalize(singleChannel, out8U, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);
    }
    return out8U;
}

cv::Mat MsgfAlgorithm::selfGuidedFilter(const cv::Mat& I, int radius, double eps)
{
    if (I.empty() || I.channels() != 1) {
        throw std::invalid_argument("selfGuidedFilter requires a single-channel image");
    }

    cv::Mat src32F;
    if (I.depth() == CV_32F) {
        src32F = I;
    } else {
        I.convertTo(src32F, CV_32F);
    }

    // Per Section 2.1: when eps == 0 (or radius <= 0), output equals input with no filtering.
    if (radius <= 0 || eps <= 0.0) {
        return src32F.clone();
    }

    const int ksize = 2 * radius + 1;
    const cv::Size win(ksize, ksize);

    cv::Mat meanI;
    cv::Mat meanII;
    cv::boxFilter(src32F, meanI, CV_32F, win, cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    cv::boxFilter(src32F.mul(src32F), meanII, CV_32F, win, cv::Point(-1, -1), true, cv::BORDER_REFLECT);

    // Local variance sigma_k^2 = E[I^2] - (E[I])^2, clamped >= 0 against fp roundoff.
    cv::Mat varI = meanII - meanI.mul(meanI);
    cv::max(varI, 0.0f, varI);

    // Eq. (8): a_k = sigma_k^2 / (sigma_k^2 + eps)
    cv::Mat a = varI / (varI + static_cast<float>(eps));
    // Eq. (9): b_k = (1 - a_k) * mu_k
    cv::Mat b = (1.0f - a).mul(meanI);

    // Eq. (7): q_i = mean(a_k) * I_i + mean(b_k)
    cv::Mat meanA;
    cv::Mat meanB;
    cv::boxFilter(a, meanA, CV_32F, win, cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    cv::boxFilter(b, meanB, CV_32F, win, cv::Point(-1, -1), true, cv::BORDER_REFLECT);

    return meanA.mul(src32F) + meanB;
}

cv::Mat MsgfAlgorithm::dynamicLinearEnhance(const cv::Mat& detailLayer, double alpha)
{
    if (detailLayer.empty()) {
        return {};
    }
    // Eq. (16): mu = (1/N) * sum(D_j)
    const float mu = static_cast<float>(cv::mean(detailLayer)[0]);
    // Eq. (17): D_adj(x, y) = alpha * (D_j(x, y) - mu) + mu
    return static_cast<float>(alpha) * (detailLayer - mu) + mu;
}

MsgfOutputs MsgfAlgorithm::run(const cv::Mat& inputGrayOrBgr, const MsgfParams& params)
{
    const auto t0 = std::chrono::steady_clock::now();
    MsgfOutputs out;

    // 1. Normalize input image to 8-bit grayscale and [0, 1] float
    out.input8U = toGray8U(inputGrayOrBgr);

    cv::Mat I;
    out.input8U.convertTo(I, CV_32F, 1.0 / 255.0);

    // 2. 3-stage cascaded Self-Guided Filter decomposition (Section 2.1, Fig. 1-2)
    out.B1 = selfGuidedFilter(I,      params.r1, params.eps1);
    out.D1 = I - out.B1;

    out.B2 = selfGuidedFilter(out.B1, params.r2, params.eps2);
    out.D2 = out.B1 - out.B2;

    out.B3 = selfGuidedFilter(out.B2, params.r3, params.eps3);
    out.D3 = out.B2 - out.B3;

    // 3. Base layer enhancement via CLAHE (Section 2.2, Eq. 10-15)
    cv::Mat B3_clamped;
    cv::min(cv::max(out.B3, 0.0f), 1.0f, B3_clamped);
    cv::Mat B3_8U;
    B3_clamped.convertTo(B3_8U, CV_8U, 255.0);

    const int gridX = std::max(1, std::min(params.claheTileGrid, B3_8U.cols));
    const int gridY = std::max(1, std::min(params.claheTileGrid, B3_8U.rows));
    auto clahe = cv::createCLAHE(
        std::max(0.0, params.claheClipLimit),
        cv::Size(gridX, gridY)
    );
    cv::Mat IB_8U;
    clahe->apply(B3_8U, IB_8U);
    IB_8U.convertTo(out.IB_clahe, CV_32F, 1.0 / 255.0);

    // 4. Multi-scale detail layer Dynamic Linear Enhancement (Section 2.3, Eq. 16-17)
    out.D1_enh = dynamicLinearEnhance(out.D1, params.alpha1);
    out.D2_enh = dynamicLinearEnhance(out.D2, params.alpha2);
    out.D3_enh = dynamicLinearEnhance(out.D3, params.alpha3);

    // 5. Weighted linear fusion (Section 2.4, Fig. 1, Eq. 18)
    out.ID_fused = static_cast<float>(params.w1) * out.D1_enh
                 + static_cast<float>(params.w2) * out.D2_enh
                 + static_cast<float>(params.w3) * out.D3_enh;

    cv::Mat fused = static_cast<float>(params.p) * out.IB_clahe
                  + static_cast<float>(params.q) * out.ID_fused;

    cv::min(cv::max(fused, 0.0f), 1.0f, fused);
    fused.convertTo(out.output8U, CV_8U, 255.0);

    const auto t1 = std::chrono::steady_clock::now();
    out.elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return out;
}

cv::Mat MsgfAlgorithm::normalizeForDisplay(const cv::Mat& layer32F)
{
    if (layer32F.empty()) {
        return {};
    }
    cv::Mat disp;
    cv::normalize(layer32F, disp, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);
    return disp;
}

} // namespace msgf
