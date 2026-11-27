#include "core/N5Algorithm.h"

#include "core/MsgfAlgorithm.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace msgf {
namespace {

struct GuidedBaseResult {
    cv::Mat image;
    double regularization = 0.0;
};

void validateStrictUint8ImageGeometry(const cv::Mat& image)
{
    if (image.empty()) {
        throw std::invalid_argument("thermal input must not be empty");
    }
    if (image.depth() != CV_8U) {
        throw std::invalid_argument("thermal input must use uint8");
    }
    if (image.dims != 2) {
        throw std::invalid_argument("thermal input must have 2D image geometry");
    }
}

void validateStrictN5Plane(const cv::Mat& plane)
{
    if (plane.empty() || plane.dims != 2 || plane.type() != CV_8UC1) {
        throw std::invalid_argument("N5 requires a nonempty 2D CV_8UC1 thermal plane");
    }
}

cv::Mat normalizeUint8To64F(const cv::Mat& input)
{
    cv::Mat output(input.rows, input.cols, CV_64FC1);
    for (int row = 0; row < input.rows; ++row) {
        const auto* source = input.ptr<unsigned char>(row);
        auto* destination = output.ptr<double>(row);
        for (int col = 0; col < input.cols; ++col) {
            destination[col] = static_cast<double>(source[col]) / 255.0;
        }
    }
    return output;
}

// Match np.rint (ties to even) for nonnegative clipped [0, 1] -> [0, 255].
unsigned char quantizeValueTiesToEven(double value)
{
    if (!std::isfinite(value)) {
        throw std::runtime_error("N5 reconstruction contains a nonfinite value");
    }
    const double scaled = std::clamp(value, 0.0, 1.0) * 255.0;
    const double lower = std::floor(scaled);
    const double fraction = scaled - lower;

    int rounded = static_cast<int>(lower);
    if (fraction > 0.5 || (fraction == 0.5 && rounded % 2 != 0)) {
        ++rounded;
    }
    return static_cast<unsigned char>(rounded);
}

cv::Mat quantize64FToUint8(const cv::Mat& input)
{
    cv::Mat output(input.rows, input.cols, CV_8UC1);
    for (int row = 0; row < input.rows; ++row) {
        const auto* source = input.ptr<double>(row);
        auto* destination = output.ptr<unsigned char>(row);
        for (int col = 0; col < input.cols; ++col) {
            destination[col] = quantizeValueTiesToEven(source[col]);
        }
    }
    return output;
}

cv::Mat boxMean64F(const cv::Mat& input, int window)
{
    cv::Mat output;
    cv::boxFilter(
        input,
        output,
        CV_64F,
        cv::Size(window, window),
        cv::Point(-1, -1),
        true,
        cv::BORDER_REFLECT_101
    );
    return output;
}

GuidedBaseResult guidedBase64F(const cv::Mat& source, int window)
{
    const int safeWindow = std::max(1, window);
    const cv::Mat mean = boxMean64F(source, safeWindow);

    cv::Mat squared(source.rows, source.cols, CV_64FC1);
    for (int row = 0; row < source.rows; ++row) {
        const auto* input = source.ptr<double>(row);
        auto* output = squared.ptr<double>(row);
        for (int col = 0; col < source.cols; ++col) {
            output[col] = input[col] * input[col];
        }
    }

    const cv::Mat meanSquared = boxMean64F(squared, safeWindow);
    cv::Mat variance(source.rows, source.cols, CV_64FC1);
    for (int row = 0; row < source.rows; ++row) {
        const auto* mu = mean.ptr<double>(row);
        const auto* avgSq = meanSquared.ptr<double>(row);
        auto* output = variance.ptr<double>(row);
        for (int col = 0; col < source.cols; ++col) {
            output[col] = std::max(avgSq[col] - mu[col] * mu[col], 0.0);
        }
    }

    const double epsilon = cv::mean(variance)[0];
    if (!std::isfinite(epsilon)) {
        throw std::runtime_error("N5 guided regularization is nonfinite");
    }

    cv::Mat coefficient(source.rows, source.cols, CV_64FC1);
    cv::Mat intercept(source.rows, source.cols, CV_64FC1);
    for (int row = 0; row < source.rows; ++row) {
        const auto* localVar = variance.ptr<double>(row);
        const auto* mu = mean.ptr<double>(row);
        auto* a = coefficient.ptr<double>(row);
        auto* b = intercept.ptr<double>(row);
        for (int col = 0; col < source.cols; ++col) {
            const double denom = localVar[col] + epsilon;
            a[col] = (denom > 0.0) ? (localVar[col] / denom) : 0.0;
            b[col] = (1.0 - a[col]) * mu[col];
        }
    }

    const cv::Mat avgA = boxMean64F(coefficient, safeWindow);
    const cv::Mat avgB = boxMean64F(intercept, safeWindow);

    cv::Mat base(source.rows, source.cols, CV_64FC1);
    for (int row = 0; row < source.rows; ++row) {
        const auto* input = source.ptr<double>(row);
        const auto* a = avgA.ptr<double>(row);
        const auto* b = avgB.ptr<double>(row);
        auto* output = base.ptr<double>(row);
        for (int col = 0; col < source.cols; ++col) {
            output[col] = a[col] * input[col] + b[col];
        }
    }

    return {std::move(base), epsilon};
}

N5Outputs executeN5Pipeline(const cv::Mat& plane8U, const N5Params& params, cv::Ptr<cv::CLAHE> clahe)
{
    const auto t0 = std::chrono::steady_clock::now();
    validateStrictN5Plane(plane8U);

    N5Outputs out;
    out.input8U = plane8U.clone();

    double minimum = 0.0;
    double maximum = 0.0;
    cv::minMaxLoc(out.input8U, &minimum, &maximum);
    out.constantPreserved = (minimum == maximum);

    if (out.constantPreserved) {
        const cv::Mat source = normalizeUint8To64F(out.input8U);
        out.B1 = source.clone();
        out.R1 = cv::Mat::zeros(out.input8U.size(), CV_64FC1);
        out.C_clahe8U = out.input8U.clone();
        out.U_stage18U = out.input8U.clone();
    } else {
        const cv::Mat source = normalizeUint8To64F(out.input8U);
        GuidedBaseResult decomp = guidedBase64F(source, params.stage1Window);
        out.B1 = std::move(decomp.image);
        out.baseRegularization = decomp.regularization;

        const cv::Mat baseUint8 = quantize64FToUint8(out.B1);
        clahe->apply(baseUint8, out.C_clahe8U);

        const cv::Mat mapped = normalizeUint8To64F(out.C_clahe8U);
        out.R1 = cv::Mat(out.input8U.rows, out.input8U.cols, CV_64FC1);
        cv::Mat reconstructed(out.input8U.rows, out.input8U.cols, CV_64FC1);

        for (int row = 0; row < out.input8U.rows; ++row) {
            const auto* input = source.ptr<double>(row);
            const auto* base = out.B1.ptr<double>(row);
            const auto* mappedBase = mapped.ptr<double>(row);
            auto* r1 = out.R1.ptr<double>(row);
            auto* rec = reconstructed.ptr<double>(row);
            for (int col = 0; col < out.input8U.cols; ++col) {
                const double residual = input[col] - base[col];
                r1[col] = residual;
                rec[col] = mappedBase[col] + residual;
            }
        }

        // Required intermediate UNIT uint8 boundary
        out.U_stage18U = quantize64FToUint8(reconstructed);
    }

    const cv::Mat unit = normalizeUint8To64F(out.U_stage18U);
    GuidedBaseResult small = guidedBase64F(unit, params.stage2Window);
    out.B2 = std::move(small.image);
    out.smallRegularization = small.regularization;

    out.R2 = cv::Mat(out.input8U.rows, out.input8U.cols, CV_64FC1);
    cv::Mat final64(out.input8U.rows, out.input8U.cols, CV_64FC1);
    const double att = params.postHalfAttenuation;

    for (int row = 0; row < out.input8U.rows; ++row) {
        const auto* u = unit.ptr<double>(row);
        const auto* b2 = out.B2.ptr<double>(row);
        auto* r2 = out.R2.ptr<double>(row);
        auto* dst = final64.ptr<double>(row);
        for (int col = 0; col < out.input8U.cols; ++col) {
            const double residual = u[col] - b2[col];
            r2[col] = residual;
            dst[col] = u[col] - att * residual;
        }
    }

    out.output8U = quantize64FToUint8(final64);
    if (out.constantPreserved) {
        out.output8U = out.input8U.clone();
    }

    const auto t1 = std::chrono::steady_clock::now();
    out.elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return out;
}

} // namespace

N5Outputs N5Algorithm::run(const cv::Mat& inputGrayOrBgr, const N5Params& params)
{
    const cv::Mat plane8U = MsgfAlgorithm::toGray8U(inputGrayOrBgr);
    const int grid = std::max(1, params.claheTileGrid);
    auto clahe = cv::createCLAHE(std::max(0.0, params.claheClipLimit), cv::Size(grid, grid));
    return executeN5Pipeline(plane8U, params, clahe);
}

cv::Mat N5Algorithm::normalizeForDisplay(const cv::Mat& layer64F)
{
    if (layer64F.empty()) {
        return {};
    }
    cv::Mat disp;
    cv::normalize(layer64F, disp, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);
    return disp;
}

} // namespace msgf

namespace qart::core::enhancement {

cv::Mat prepareThermalUint8Plane(const cv::Mat& image)
{
    msgf::validateStrictUint8ImageGeometry(image);
    switch (image.channels()) {
    case 1:
        return image.clone();
    case 3: {
        cv::Mat output;
        cv::cvtColor(image, output, cv::COLOR_BGR2GRAY);
        return output;
    }
    case 4: {
        cv::Mat output;
        cv::cvtColor(image, output, cv::COLOR_BGRA2GRAY);
        return output;
    }
    default:
        throw std::invalid_argument("thermal input must have 1, 3 or 4 channels");
    }
}

cv::Mat bypassThermal(const cv::Mat& plane)
{
    msgf::validateStrictUint8ImageGeometry(plane);
    if (plane.channels() != 1) {
        throw std::invalid_argument("BYPASS requires a single-channel thermal plane");
    }
    return plane.clone();
}

N5ThermalEnhancer::N5ThermalEnhancer()
    : clahe_(cv::createCLAHE(2.0, cv::Size(8, 8)))
{
}

N5EnhancementResult N5ThermalEnhancer::enhance(const cv::Mat& plane)
{
    msgf::validateStrictN5Plane(plane);
    msgf::N5Outputs out = msgf::executeN5Pipeline(plane, msgf::N5Params{}, clahe_);
    return {
        std::move(out.output8U),
        out.smallRegularization,
        out.constantPreserved
    };
}

} // namespace qart::core::enhancement
