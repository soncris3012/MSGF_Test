#include "core/BenchmarkExport.h"
#include "core/BenchmarkTypes.h"
#include "core/MsgfAlgorithm.h"
#include "core/MsgfBaselines.h"
#include "core/MsgfMetrics.h"
#include "core/MsgfParams.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool cond, const std::string& msg)
{
    if (!cond) {
        throw std::runtime_error("Test failed: " + msg);
    }
}

template <typename Fn>
void requireThrowsInvalid(Fn&& fn, const std::string& msg)
{
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Expected std::invalid_argument: " + msg);
}

void testSelfGuidedFilterIdentityAndEdgePreservation()
{
    cv::Mat step(64, 64, CV_32FC1, cv::Scalar(0.2f));
    step(cv::Rect(32, 0, 32, 64)).setTo(0.8f);

    // eps == 0 or radius == 0 -> exact identity (Section 2.1)
    const cv::Mat qZeroEps = MsgfAlgorithm::selfGuidedFilter(step, 4, 0.0);
    require(cv::norm(qZeroEps - step, cv::NORM_INF) < 1e-6,
            "selfGuidedFilter(eps=0) should equal input");

    const cv::Mat qZeroRad = MsgfAlgorithm::selfGuidedFilter(step, 0, 0.05);
    require(cv::norm(qZeroRad - step, cv::NORM_INF) < 1e-6,
            "selfGuidedFilter(radius=0) should equal input");

    // With eps > 0, strong step edge remains sharp while flat regions stay smooth
    const cv::Mat qFiltered = MsgfAlgorithm::selfGuidedFilter(step, 4, 0.01);
    require(qFiltered.at<float>(32, 10) < 0.25f, "left plateau drifted");
    require(qFiltered.at<float>(32, 54) > 0.75f, "right plateau drifted");
    require(qFiltered.at<float>(32, 33) - qFiltered.at<float>(32, 30) > 0.35f,
            "step edge over-smoothed");
}

void testDecompositionConservationAndSignedDetails()
{
    cv::Mat img(96, 128, CV_8UC1);
    cv::RNG rng(42);
    rng.fill(img, cv::RNG::UNIFORM, 25, 225);

    MsgfParams params;
    const MsgfOutputs out = MsgfAlgorithm::run(img, params);

    require(out.input8U.type() == CV_8UC1 && out.output8U.type() == CV_8UC1,
            "input/output types must be CV_8UC1");
    require(out.D1.type() == CV_32FC1 && out.D2.type() == CV_32FC1 && out.D3.type() == CV_32FC1,
            "detail layers must remain signed CV_32FC1");

    // Check D1, D2, D3 genuinely contain both negative and positive values
    double d1Min = 0.0, d1Max = 0.0;
    cv::minMaxLoc(out.D1, &d1Min, &d1Max);
    require(d1Min < -0.01 && d1Max > 0.01, "D1 must retain signed positive and negative edges");

    // Algebraic decomposition identity: I == B3 + D1 + D2 + D3
    cv::Mat I;
    out.input8U.convertTo(I, CV_32F, 1.0 / 255.0);
    const cv::Mat reconstructed = out.B3 + out.D1 + out.D2 + out.D3;
    const double err = cv::norm(I - reconstructed, cv::NORM_INF);
    require(err < 1e-5, "3-scale decomposition does not sum back to input I");
}

void testDynamicLinearEnhancementEq16And17()
{
    cv::Mat detail(40, 40, CV_32FC1);
    cv::RNG rng(7);
    rng.fill(detail, cv::RNG::NORMAL, 0.02, 0.10);

    const cv::Mat same = MsgfAlgorithm::dynamicLinearEnhance(detail, 1.0);
    require(cv::norm(same - detail, cv::NORM_INF) < 1e-6,
            "alpha=1.0 should preserve detail layer identically");

    const double alpha = 1.75;
    const cv::Mat enhanced = MsgfAlgorithm::dynamicLinearEnhance(detail, alpha);

    cv::Scalar muIn, stdIn, muOut, stdOut;
    cv::meanStdDev(detail, muIn, stdIn);
    cv::meanStdDev(enhanced, muOut, stdOut);

    require(std::abs(muIn[0] - muOut[0]) < 1e-5,
            "Dynamic Linear Enhancement must preserve mean mu_j (Eq. 16-17)");
    require(std::abs(stdOut[0] - alpha * stdIn[0]) < 1e-4,
            "Dynamic Linear Enhancement must scale stdDev by alpha");
}

void testMultiFormatAndTinyGeometries()
{
    MsgfParams params;

    // 16-bit thermal input
    cv::Mat u16(48, 64, CV_16UC1);
    for (int r = 0; r < u16.rows; ++r) {
        for (int c = 0; c < u16.cols; ++c) {
            u16.at<unsigned short>(r, c) = static_cast<unsigned short>(2000 + r * 400 + c * 150);
        }
    }
    const MsgfOutputs out16 = MsgfAlgorithm::run(u16, params);
    require(out16.output8U.size() == u16.size() && out16.output8U.type() == CV_8UC1,
            "16-bit thermal input handling failed");

    // BGR 3-channel input
    cv::Mat bgr(32, 32, CV_8UC3, cv::Scalar(40, 100, 180));
    const MsgfOutputs outBgr = MsgfAlgorithm::run(bgr, params);
    require(outBgr.output8U.size() == bgr.size(), "3-channel BGR input failed");

    // Tiny edge geometry (smaller than default 8x8 CLAHE grid)
    cv::Mat tiny(4, 3, CV_8UC1, cv::Scalar(120));
    const MsgfOutputs outTiny = MsgfAlgorithm::run(tiny, params);
    require(outTiny.output8U.size() == tiny.size(), "tiny image failed");

    // Empty input throws invalid_argument
    requireThrowsInvalid([&]() {
        (void)MsgfAlgorithm::run(cv::Mat(), params);
    }, "empty input");
}

void testMetricsEq19To21AndPiqe()
{
    // 1. Identical images -> PSNR 99.99, MAE 0
    cv::Mat a(64, 64, CV_8UC1, cv::Scalar(128));
    const MsgfMetricResult identical = MsgfMetrics::evaluate(a, a);
    require(std::abs(identical.psnr - 99.99) < 1e-6, "identical PSNR mismatch");
    require(std::abs(identical.ie) < 1e-6, "constant IE must be 0");
    require(std::abs(identical.ag) < 1e-6, "constant AG must be 0");
    require(std::abs(identical.piqe - 100.0) < 1e-6, "constant PIQE must be 100");

    // 2. Uniform 256-level histogram -> IE = 8.0 bits
    cv::Mat uniform256(32, 256, CV_8UC1);
    for (int r = 0; r < uniform256.rows; ++r) {
        for (int c = 0; c < uniform256.cols; ++c) {
            uniform256.at<unsigned char>(r, c) = static_cast<unsigned char>(c);
        }
    }
    const MsgfPlaneStats uStats = MsgfMetrics::evaluateSingle(uniform256);
    require(std::abs(uStats.ie - 8.0) < 1e-5, "uniform 256-level IE must be 8.0");

    // 3. Sharp texture vs heavy Gaussian blur: sharp has higher AG and lower (better) PIQE
    cv::Mat sharp(128, 128, CV_8UC1);
    for (int r = 0; r < sharp.rows; ++r) {
        for (int c = 0; c < sharp.cols; ++c) {
            const double val = 128.0 + 70.0 * std::sin(0.35 * r) * std::cos(0.35 * c)
                             + ((r % 4 == 0) ? 25.0 : -15.0);
            sharp.at<unsigned char>(r, c) = static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
        }
    }
    cv::Mat blurred;
    cv::GaussianBlur(sharp, blurred, cv::Size(15, 15), 4.0);

    const MsgfPlaneStats sSharp = MsgfMetrics::evaluateSingle(sharp);
    const MsgfPlaneStats sBlur = MsgfMetrics::evaluateSingle(blurred);
    require(sSharp.ag > sBlur.ag, "sharp image must have strictly higher AG than blurred image");
    require(sBlur.piqe > sSharp.piqe, "blurred image must have worse (higher) PIQE than sharp image");
}

void testSampleImageSmokeIfAvailable()
{
    namespace fs = std::filesystem;
    const fs::path candidates[] = {
        "sample_images/scene1_original.png",
        "../sample_images/scene1_original.png",
        "../../sample_images/scene1_original.png"
    };
    fs::path found;
    for (const auto& p : candidates) {
        if (fs::exists(p)) {
            found = p;
            break;
        }
    }
    if (found.empty()) {
        std::cout << "[INFO] sample_images not found relative to CWD; skipping file smoke test.\n";
        return;
    }

    const cv::Mat img = cv::imread(found.string(), cv::IMREAD_UNCHANGED);
    require(!img.empty(), "failed to load scene1_original.png");

    const MsgfOutputs out = MsgfAlgorithm::run(img, MsgfParams{});
    const MsgfMetricResult m = MsgfMetrics::evaluate(out.input8U, out.output8U);
    require(m.valid, "metric result invalid on scene1");
    require(m.psnr > 10.0 && m.psnr < 99.0, "unexpected PSNR on scene1");
    require(m.ag > m.inputStats.ag, "MSGF-TIR should increase AG on scene1");
    require(m.ie > m.inputStats.ie, "MSGF-TIR should increase IE on scene1");
}

void testBaselinesSummaryAndExport()
{
    cv::Mat sample(64, 64, CV_8UC1);
    for (int r = 0; r < sample.rows; ++r) {
        for (int c = 0; c < sample.cols; ++c) {
            sample.at<unsigned char>(r, c) = static_cast<unsigned char>(60 + ((r * 3 + c * 5) % 120));
        }
    }

    const MsgfParams params;
    const std::vector<msgf::BenchmarkAlgorithmId> algos = {
        msgf::BenchmarkAlgorithmId::Original,
        msgf::BenchmarkAlgorithmId::HE,
        msgf::BenchmarkAlgorithmId::GF_DDE,
        msgf::BenchmarkAlgorithmId::BF_DDE,
        msgf::BenchmarkAlgorithmId::MSGF_TIR
    };

    std::map<msgf::BenchmarkAlgorithmId, std::vector<msgf::ImageBenchmarkRow>> perAlgo;
    const MsgfPlaneStats origStats = MsgfMetrics::evaluateSingle(sample);
    for (int idx = 1; idx <= 2; ++idx) {
        for (msgf::BenchmarkAlgorithmId id : algos) {
            const auto res = msgf::MsgfBaselines::runAlgorithm(id, sample, params);
            require(res.output8U.size() == sample.size() && res.output8U.type() == CV_8UC1,
                    "baseline output type/size mismatch");
            const MsgfMetricResult m = MsgfMetrics::evaluate(sample, res.output8U);

            msgf::ImageBenchmarkRow row;
            row.index = idx;
            row.fileName = "img_" + std::to_string(idx) + ".png";
            row.fullPath = "/tmp/" + row.fileName;
            row.width = sample.cols;
            row.height = sample.rows;
            row.origIE = origStats.ie;
            row.origAG = origStats.ag;
            row.origPIQE = origStats.piqe;
            row.psnr = m.psnr;
            row.ie = m.ie;
            row.ag = m.ag;
            row.piqe = m.piqe;
            row.runtimeMs = res.elapsedMs;
            perAlgo[id].push_back(row);
        }
    }

    const auto summaries = msgf::summarizeDatasetBenchmark(algos, perAlgo);
    require(summaries.size() == algos.size(), "summary row count mismatch");

    double winSum = 0.0;
    for (const auto& s : summaries) {
        if (s.algoId != msgf::BenchmarkAlgorithmId::Original) {
            require(s.count == 2, "summary sample count mismatch");
            require(s.winRatePct >= 0.0 && s.winRatePct <= 100.0, "winRate out of range");
            winSum += s.winRatePct;
        }
    }
    require(winSum >= 99.0, "at least one method must earn rank-1 wins");

    const std::string sumCsv = msgf::exportSummaryToCsvUtf8Bom(summaries);
    const std::string detCsv = msgf::exportDetailToCsvUtf8Bom(algos, perAlgo);
    require(sumCsv.size() >= 3 && sumCsv[0] == '\xEF' && sumCsv[1] == '\xBB' && sumCsv[2] == '\xBF',
            "summary CSV missing UTF-8 BOM");
    require(detCsv.size() >= 3 && detCsv[0] == '\xEF' && detCsv[1] == '\xBB' && detCsv[2] == '\xBF',
            "detail CSV missing UTF-8 BOM");

    const std::string tex = msgf::exportSummaryToLatex(summaries);
    require(tex.find("\\textbf{") != std::string::npos, "LaTeX export missing bold highlight");
    require(tex.find("\\underline{") != std::string::npos, "LaTeX export missing underline 2nd-best");

    const std::string md = msgf::exportSummaryToMarkdown(summaries);
    require(md.find("**") != std::string::npos, "Markdown export missing bold highlight");
}

} // namespace

int main()
{
    try {
        testSelfGuidedFilterIdentityAndEdgePreservation();
        testDecompositionConservationAndSignedDetails();
        testDynamicLinearEnhancementEq16And17();
        testMultiFormatAndTinyGeometries();
        testMetricsEq19To21AndPiqe();
        testSampleImageSmokeIfAvailable();
        testBaselinesSummaryAndExport();
        std::cout << "All MSGF-TIR unit & invariant tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
