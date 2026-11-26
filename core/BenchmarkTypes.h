#pragma once

#include "core/MsgfBaselines.h"
#include "core/MsgfMetrics.h"

#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace msgf {

/**
 * @brief Per-image metrics recorded for one algorithm on one dataset image.
 */
struct ImageBenchmarkRow {
    int index = 0; // 1-based display index
    std::string fileName;
    std::string fullPath;
    int width = 0;
    int height = 0;

    double origIE = 0.0;
    double origAG = 0.0;
    double origPIQE = 0.0;

    double psnr = 0.0;
    double ie = 0.0;
    double ag = 0.0;
    double piqe = 0.0;
    double runtimeMs = 0.0;
};

/**
 * @brief Aggregate Mean +/- StdDev + Win Rate for one algorithm across a dataset.
 */
struct AlgoSummaryRow {
    BenchmarkAlgorithmId algoId = BenchmarkAlgorithmId::Original;
    std::string algoName;
    int count = 0;

    double meanPsnr = 0.0, stdPsnr = 0.0;
    double meanIE   = 0.0, stdIE   = 0.0;
    double meanAG   = 0.0, stdAG   = 0.0;
    double meanPIQE = 0.0, stdPIQE = 0.0;
    double meanMs   = 0.0, stdMs   = 0.0;

    // Percentage [0, 100] of images where this method achieves rank-1 score vs competitors
    double winRatePct = 0.0;
    // Detailed win-rate of MSGF-TIR (or this algo) per indicator
    double winRateIEPct = 0.0;
    double winRateAGPct = 0.0;
    double winRatePIQEPct = 0.0;
    double winRatePSNRPct = 0.0;
};

struct MeanStdResult {
    double mean = 0.0;
    double stdDev = 0.0;
};

inline MeanStdResult computeMeanStd(const std::vector<double>& vals)
{
    if (vals.empty()) {
        return {};
    }
    double sum = 0.0;
    for (double v : vals) {
        sum += v;
    }
    const double mean = sum / static_cast<double>(vals.size());
    double sqSum = 0.0;
    for (double v : vals) {
        const double d = v - mean;
        sqSum += d * d;
    }
    const double variance = sqSum / static_cast<double>(vals.size());
    return {mean, std::sqrt(std::max(0.0, variance))};
}

/**
 * @brief Compute summary Mean +/- StdDev and Win Rates from collected per-algorithm rows.
 */
inline std::vector<AlgoSummaryRow> summarizeDatasetBenchmark(
    const std::vector<BenchmarkAlgorithmId>& orderedAlgos,
    const std::map<BenchmarkAlgorithmId, std::vector<ImageBenchmarkRow>>& perAlgoRows)
{
    std::vector<AlgoSummaryRow> summaries;
    if (orderedAlgos.empty()) {
        return summaries;
    }

    // Determine common image count across evaluated algorithms
    std::size_t nImages = 0;
    std::vector<BenchmarkAlgorithmId> competitors;
    for (BenchmarkAlgorithmId id : orderedAlgos) {
        auto it = perAlgoRows.find(id);
        if (it == perAlgoRows.end() || it->second.empty()) {
            continue;
        }
        if (nImages == 0) {
            nImages = it->second.size();
        } else {
            nImages = std::min(nImages, it->second.size());
        }
        if (id != BenchmarkAlgorithmId::Original) {
            competitors.push_back(id);
        }
    }

    for (BenchmarkAlgorithmId id : orderedAlgos) {
        auto it = perAlgoRows.find(id);
        if (it == perAlgoRows.end() || it->second.empty()) {
            continue;
        }
        const auto& rows = it->second;
        const std::size_t n = rows.size();

        std::vector<double> psnrs, ies, ags, piqes, mss;
        psnrs.reserve(n);
        ies.reserve(n);
        ags.reserve(n);
        piqes.reserve(n);
        mss.reserve(n);

        for (const auto& r : rows) {
            psnrs.push_back(r.psnr);
            ies.push_back(r.ie);
            ags.push_back(r.ag);
            piqes.push_back(r.piqe);
            mss.push_back(r.runtimeMs);
        }

        AlgoSummaryRow s;
        s.algoId = id;
        s.algoName = MsgfBaselines::algorithmName(id);
        s.count = static_cast<int>(n);

        const auto stPsnr = computeMeanStd(psnrs);
        const auto stIe   = computeMeanStd(ies);
        const auto stAg   = computeMeanStd(ags);
        const auto stPiqe = computeMeanStd(piqes);
        const auto stMs   = computeMeanStd(mss);

        s.meanPsnr = stPsnr.mean; s.stdPsnr = stPsnr.stdDev;
        s.meanIE   = stIe.mean;   s.stdIE   = stIe.stdDev;
        s.meanAG   = stAg.mean;   s.stdAG   = stAg.stdDev;
        s.meanPIQE = stPiqe.mean; s.stdPIQE = stPiqe.stdDev;
        s.meanMs   = stMs.mean;   s.stdMs   = stMs.stdDev;

        // Compute per-image rank-1 win rates among enhanced methods (excluding Original)
        if (id != BenchmarkAlgorithmId::Original && nImages > 0 && !competitors.empty()) {
            int winPsnr = 0, winIE = 0, winAG = 0, winPIQE = 0, winPoints = 0;
            for (std::size_t i = 0; i < nImages; ++i) {
                const auto& cur = rows[i];
                bool bestPsnr = true, bestIE = true, bestAG = true, bestPIQE = true;
                for (BenchmarkAlgorithmId oppId : competitors) {
                    if (oppId == id) continue;
                    const auto& opp = perAlgoRows.at(oppId)[i];
                    if (opp.psnr > cur.psnr) bestPsnr = false;
                    if (opp.ie   > cur.ie)   bestIE   = false;
                    if (opp.ag   > cur.ag)   bestAG   = false;
                    if (opp.piqe < cur.piqe) bestPIQE = false;
                }
                if (bestPsnr) winPsnr++;
                if (bestIE)   winIE++;
                if (bestAG)   winAG++;
                if (bestPIQE) winPIQE++;
                winPoints += (bestPsnr ? 1 : 0) + (bestIE ? 1 : 0)
                           + (bestAG ? 1 : 0) + (bestPIQE ? 1 : 0);
            }
            const double denom = static_cast<double>(nImages);
            s.winRatePSNRPct = 100.0 * static_cast<double>(winPsnr) / denom;
            s.winRateIEPct   = 100.0 * static_cast<double>(winIE)   / denom;
            s.winRateAGPct   = 100.0 * static_cast<double>(winAG)   / denom;
            s.winRatePIQEPct = 100.0 * static_cast<double>(winPIQE) / denom;
            s.winRatePct     = 100.0 * static_cast<double>(winPoints) / (4.0 * denom);
        }

        summaries.push_back(std::move(s));
    }

    return summaries;
}

} // namespace msgf
