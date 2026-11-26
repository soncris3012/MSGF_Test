#include "core/MsgfMetrics.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace msgf {
namespace {

double computeShannonEntropy(const cv::Mat& gray8U)
{
    std::array<std::size_t, 256> hist{};
    for (int r = 0; r < gray8U.rows; ++r) {
        const auto* ptr = gray8U.ptr<unsigned char>(r);
        for (int c = 0; c < gray8U.cols; ++c) {
            hist[ptr[c]]++;
        }
    }

    const double total = static_cast<double>(gray8U.total());
    if (total <= 0.0) {
        return 0.0;
    }

    double entropy = 0.0;
    for (std::size_t count : hist) {
        if (count > 0) {
            const double p = static_cast<double>(count) / total;
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

double computeAverageGradient(const cv::Mat& gray8U)
{
    if (gray8U.rows < 2 || gray8U.cols < 2) {
        return 0.0;
    }

    double sumG = 0.0;
    for (int r = 0; r < gray8U.rows - 1; ++r) {
        const auto* row0 = gray8U.ptr<unsigned char>(r);
        const auto* row1 = gray8U.ptr<unsigned char>(r + 1);
        for (int c = 0; c < gray8U.cols - 1; ++c) {
            const double cur = static_cast<double>(row0[c]);
            const double dx = static_cast<double>(row1[c]) - cur;
            const double dy = static_cast<double>(row0[c + 1]) - cur;
            sumG += std::sqrt((dx * dx + dy * dy) / 2.0);
        }
    }

    const double denom =
        static_cast<double>(gray8U.rows - 1) * static_cast<double>(gray8U.cols - 1);
    return denom > 0.0 ? (sumG / denom) : 0.0;
}

struct RegionStats {
    double mean = 0.0;
    double stdDev = 0.0;
    double variance = 0.0;
};

RegionStats regionStats2D(const double* topLeft, int stride, int rows, int cols)
{
    const int n = rows * cols;
    if (n <= 0) {
        return {};
    }
    double sum = 0.0;
    for (int r = 0; r < rows; ++r) {
        const double* rowPtr = topLeft + r * stride;
        for (int c = 0; c < cols; ++c) {
            sum += rowPtr[c];
        }
    }
    const double mean = sum / static_cast<double>(n);

    double sqSum = 0.0;
    for (int r = 0; r < rows; ++r) {
        const double* rowPtr = topLeft + r * stride;
        for (int c = 0; c < cols; ++c) {
            const double d = rowPtr[c] - mean;
            sqSum += d * d;
        }
    }
    const double variance = std::max(0.0, sqSum / static_cast<double>(n));
    return {mean, std::sqrt(variance), variance};
}

double variance1D(const double* data, int stride, int len)
{
    if (len <= 0) {
        return 0.0;
    }
    double sum = 0.0;
    for (int i = 0; i < len; ++i) {
        sum += data[i * stride];
    }
    const double mean = sum / static_cast<double>(len);
    double sqSum = 0.0;
    for (int i = 0; i < len; ++i) {
        const double d = data[i * stride] - mean;
        sqSum += d * d;
    }
    return sqSum / static_cast<double>(len);
}

// Perception-based Image Quality Evaluator (PIQE, Venkatanath et al., NCC 2015).
// Lower score indicates better perceptual quality; output range (0, 100].
double computePiqe(const cv::Mat& gray8U)
{
    constexpr int kBlockSize = 16;
    constexpr int kSegLen = 6;
    constexpr double kActivityThreshold = 0.1;
    constexpr double kStructVarThreshold = 0.18;
    constexpr double kNoiseVarThreshold = 0.45;
    constexpr double kPoolingConst = 1.0;

    if (gray8U.rows < kBlockSize || gray8U.cols < kBlockSize) {
        return 100.0;
    }

    cv::Mat gray64;
    gray8U.convertTo(gray64, CV_64F);

    cv::Mat mu;
    cv::GaussianBlur(
        gray64,
        mu,
        cv::Size(7, 7),
        7.0 / 6.0,
        7.0 / 6.0,
        cv::BORDER_REPLICATE
    );

    cv::Mat meanSq;
    cv::GaussianBlur(
        gray64.mul(gray64),
        meanSq,
        cv::Size(7, 7),
        7.0 / 6.0,
        7.0 / 6.0,
        cv::BORDER_REPLICATE
    );

    cv::Mat varMap = meanSq - mu.mul(mu);
    cv::max(varMap, 0.0, varMap);
    cv::Mat sigma;
    cv::sqrt(varMap, sigma);

    cv::Mat mscn = (gray64 - mu) / (sigma + 1.0);

    const int blockRows = mscn.rows / kBlockSize;
    const int blockCols = mscn.cols / kBlockSize;
    const int mscnStep = static_cast<int>(mscn.step / sizeof(double));

    int activeBlocks = 0;
    double distortionSum = 0.0;

    for (int br = 0; br < blockRows; ++br) {
        const double* rowBlockBase = mscn.ptr<double>(br * kBlockSize);
        for (int bc = 0; bc < blockCols; ++bc) {
            const double* blockPtr = rowBlockBase + bc * kBlockSize;
            const RegionStats blkStats =
                regionStats2D(blockPtr, mscnStep, kBlockSize, kBlockSize);
            const double blockStd = blkStats.stdDev;
            const double blockVar = blkStats.variance;

            if (blockVar <= kActivityThreshold) {
                continue;
            }
            activeBlocks++;

            int lowVarSegCount = 0;
            const int segStarts = kBlockSize - kSegLen + 1; // 11
            for (int s = 0; s < segStarts; ++s) {
                if (variance1D(blockPtr + s, 1, kSegLen) < kActivityThreshold) {
                    lowVarSegCount++;
                }
                if (variance1D(blockPtr + 15 * mscnStep + s, 1, kSegLen) < kActivityThreshold) {
                    lowVarSegCount++;
                }
                if (variance1D(blockPtr + s * mscnStep, mscnStep, kSegLen) < kActivityThreshold) {
                    lowVarSegCount++;
                }
                if (variance1D(blockPtr + s * mscnStep + 15, mscnStep, kSegLen) < kActivityThreshold) {
                    lowVarSegCount++;
                }
            }

            const double stdCol7 =
                std::sqrt(std::max(0.0, variance1D(blockPtr + 7, mscnStep, kBlockSize)));
            const double stdCol8 =
                std::sqrt(std::max(0.0, variance1D(blockPtr + 8, mscnStep, kBlockSize)));
            const double centerColRatio =
                (blockStd > 1e-12) ? (0.5 * (stdCol7 + stdCol8) / blockStd) : 0.0;

            const RegionStats interiorStats =
                regionStats2D(blockPtr + 3 * mscnStep + 3, mscnStep, 10, 10);
            const double interiorVarRatio =
                (blockVar > 1e-12) ? (interiorStats.variance / blockVar) : 0.0;

            double dStruct = 0.0;
            if (blockVar < kStructVarThreshold
                || lowVarSegCount >= 18
                || (centerColRatio > 1.25 && lowVarSegCount >= 4)) {
                dStruct = std::clamp(1.0 - blockStd, 0.0, 1.0);
            }

            double dNoise = 0.0;
            if (blockVar > kNoiseVarThreshold
                && interiorVarRatio > 0.78
                && lowVarSegCount <= 2) {
                dNoise = std::clamp(blockStd, 0.0, 1.0);
            }

            distortionSum += std::max(dStruct, dNoise);
        }
    }

    if (activeBlocks == 0) {
        return 100.0;
    }

    const double rawScore =
        ((distortionSum + kPoolingConst)
         / (static_cast<double>(activeBlocks) + kPoolingConst))
        * 100.0;
    return std::clamp(rawScore, 0.0, 100.0);
}

} // namespace

MsgfPlaneStats MsgfMetrics::evaluateSingle(const cv::Mat& gray8U)
{
    MsgfPlaneStats stats;
    if (gray8U.empty() || gray8U.dims != 2 || gray8U.type() != CV_8UC1) {
        return stats;
    }

    cv::Scalar meanScalar;
    cv::Scalar stdScalar;
    cv::meanStdDev(gray8U, meanScalar, stdScalar);

    stats.valid = true;
    stats.mean = meanScalar[0];
    stats.stdDev = stdScalar[0];
    stats.ie = computeShannonEntropy(gray8U);
    stats.ag = computeAverageGradient(gray8U);
    stats.piqe = computePiqe(gray8U);
    return stats;
}

MsgfMetricResult MsgfMetrics::evaluate(const cv::Mat& orig8U, const cv::Mat& enh8U)
{
    MsgfMetricResult res;
    if (orig8U.empty() || enh8U.empty() || orig8U.size() != enh8U.size()
        || orig8U.type() != CV_8UC1 || enh8U.type() != CV_8UC1) {
        throw std::invalid_argument("MsgfMetrics::evaluate requires equal-sized CV_8UC1 images");
    }

    res.inputStats = evaluateSingle(orig8U);
    res.outputStats = evaluateSingle(enh8U);
    res.ie = res.outputStats.ie;
    res.ag = res.outputStats.ag;
    res.piqe = res.outputStats.piqe;

    // PSNR - Eq. (19)
    cv::Mat diff;
    cv::absdiff(orig8U, enh8U, diff);
    res.mae = cv::mean(diff)[0];

    diff.convertTo(diff, CV_64F);
    const double mse = cv::mean(diff.mul(diff))[0];
    res.psnr = (mse <= 1e-12) ? 99.99 : 10.0 * std::log10((255.0 * 255.0) / mse);
    res.valid = res.inputStats.valid && res.outputStats.valid;
    return res;
}

} // namespace msgf
