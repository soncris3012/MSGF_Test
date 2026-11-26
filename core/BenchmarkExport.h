#pragma once

#include "core/BenchmarkTypes.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace msgf {

namespace detail {

inline std::string fmtNum(double val, int prec = 2)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << val;
    return oss.str();
}

inline std::string fmtMeanStdText(double mean, double stdDev, int prec = 2)
{
    return fmtNum(mean, prec) + " ± " + fmtNum(stdDev, prec);
}

inline std::string fmtMeanStdLatex(double mean, double stdDev, int prec = 2)
{
    return fmtNum(mean, prec) + " $\\pm$ " + fmtNum(stdDev, prec);
}

struct RankPair {
    int best = -1;
    int second = -1;
};

template <typename Getter>
inline RankPair findBestAndSecond(
    const std::vector<AlgoSummaryRow>& summaries,
    Getter getter,
    bool higherIsBetter,
    bool skipOriginal = true)
{
    std::vector<std::pair<double, int>> candidates;
    for (int i = 0; i < static_cast<int>(summaries.size()); ++i) {
        if (skipOriginal && summaries[static_cast<std::size_t>(i)].algoId == BenchmarkAlgorithmId::Original) {
            continue;
        }
        candidates.emplace_back(getter(summaries[static_cast<std::size_t>(i)]), i);
    }
    if (candidates.empty()) {
        return {};
    }
    std::stable_sort(candidates.begin(), candidates.end(), [higherIsBetter](const auto& a, const auto& b) {
        return higherIsBetter ? (a.first > b.first) : (a.first < b.first);
    });
    RankPair rp;
    rp.best = candidates[0].second;
    if (candidates.size() >= 2) {
        rp.second = candidates[1].second;
    }
    return rp;
}

inline std::string csvEscape(const std::string& s)
{
    if (s.find_first_of(",\"\r\n") == std::string::npos) {
        return s;
    }
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') {
            out += "\"\"";
        } else {
            out += c;
        }
    }
    out += '"';
    return out;
}

} // namespace detail

/**
 * @brief Serialize summary table to CSV with UTF-8 BOM (\xEF\xBB\xBF) for clean Excel import.
 */
inline std::string exportSummaryToCsvUtf8Bom(const std::vector<AlgoSummaryRow>& summaries)
{
    std::ostringstream oss;
    oss << "\xEF\xBB\xBF";
    oss << "Algorithm,Images,PSNR_Mean_dB,PSNR_Std,IE_Mean,IE_Std,AG_Mean,AG_Std,"
           "PIQE_Mean,PIQE_Std,Time_Mean_ms,Time_Std,WinRate_Pct,WinRate_PSNR_Pct,"
           "WinRate_IE_Pct,WinRate_AG_Pct,WinRate_PIQE_Pct\n";
    for (const auto& s : summaries) {
        const bool isOrig = (s.algoId == BenchmarkAlgorithmId::Original);
        oss << detail::csvEscape(s.algoName) << ','
            << s.count << ','
            << (isOrig ? "" : detail::fmtNum(s.meanPsnr, 4)) << ','
            << (isOrig ? "" : detail::fmtNum(s.stdPsnr, 4)) << ','
            << detail::fmtNum(s.meanIE, 4) << ','
            << detail::fmtNum(s.stdIE, 4) << ','
            << detail::fmtNum(s.meanAG, 4) << ','
            << detail::fmtNum(s.stdAG, 4) << ','
            << detail::fmtNum(s.meanPIQE, 4) << ','
            << detail::fmtNum(s.stdPIQE, 4) << ','
            << (isOrig ? "" : detail::fmtNum(s.meanMs, 4)) << ','
            << (isOrig ? "" : detail::fmtNum(s.stdMs, 4)) << ','
            << (isOrig ? "" : detail::fmtNum(s.winRatePct, 2)) << ','
            << (isOrig ? "" : detail::fmtNum(s.winRatePSNRPct, 2)) << ','
            << (isOrig ? "" : detail::fmtNum(s.winRateIEPct, 2)) << ','
            << (isOrig ? "" : detail::fmtNum(s.winRateAGPct, 2)) << ','
            << (isOrig ? "" : detail::fmtNum(s.winRatePIQEPct, 2)) << '\n';
    }
    return oss.str();
}

/**
 * @brief Serialize per-image detail rows across all benchmarked algorithms to CSV with UTF-8 BOM.
 */
inline std::string exportDetailToCsvUtf8Bom(
    const std::vector<BenchmarkAlgorithmId>& orderedAlgos,
    const std::map<BenchmarkAlgorithmId, std::vector<ImageBenchmarkRow>>& perAlgoRows)
{
    std::ostringstream oss;
    oss << "\xEF\xBB\xBF";
    oss << "Index,Algorithm,FileName,Width,Height,PSNR_dB,IE_Orig,IE_Out,Delta_IE,"
           "AG_Orig,AG_Out,Delta_AG,PIQE_Orig,PIQE_Out,Delta_PIQE,Time_ms,FullPath\n";

    for (BenchmarkAlgorithmId id : orderedAlgos) {
        if (id == BenchmarkAlgorithmId::Original) {
            continue;
        }
        auto it = perAlgoRows.find(id);
        if (it == perAlgoRows.end()) {
            continue;
        }
        const std::string algoName = MsgfBaselines::algorithmName(id);
        for (const auto& r : it->second) {
            oss << r.index << ','
                << detail::csvEscape(algoName) << ','
                << detail::csvEscape(r.fileName) << ','
                << r.width << ','
                << r.height << ','
                << detail::fmtNum(r.psnr, 4) << ','
                << detail::fmtNum(r.origIE, 4) << ','
                << detail::fmtNum(r.ie, 4) << ','
                << detail::fmtNum(r.ie - r.origIE, 4) << ','
                << detail::fmtNum(r.origAG, 4) << ','
                << detail::fmtNum(r.ag, 4) << ','
                << detail::fmtNum(r.ag - r.origAG, 4) << ','
                << detail::fmtNum(r.origPIQE, 4) << ','
                << detail::fmtNum(r.piqe, 4) << ','
                << detail::fmtNum(r.piqe - r.origPIQE, 4) << ','
                << detail::fmtNum(r.runtimeMs, 4) << ','
                << detail::csvEscape(r.fullPath) << '\n';
        }
    }
    return oss.str();
}

/**
 * @brief Generate a publication-ready LaTeX table highlighting best (\textbf) and 2nd-best (\underline).
 */
inline std::string exportSummaryToLatex(const std::vector<AlgoSummaryRow>& summaries)
{
    const auto rPsnr = detail::findBestAndSecond(summaries, [](const auto& s) { return s.meanPsnr; }, true);
    const auto rIE   = detail::findBestAndSecond(summaries, [](const auto& s) { return s.meanIE; }, true);
    const auto rAG   = detail::findBestAndSecond(summaries, [](const auto& s) { return s.meanAG; }, true);
    const auto rPiqe = detail::findBestAndSecond(summaries, [](const auto& s) { return s.meanPIQE; }, false);
    const auto rMs   = detail::findBestAndSecond(summaries, [](const auto& s) { return s.meanMs; }, false);
    const auto rWin  = detail::findBestAndSecond(summaries, [](const auto& s) { return s.winRatePct; }, true);

    auto decorate = [](const std::string& cell, int rowIdx, const detail::RankPair& rp) -> std::string {
        if (rowIdx == rp.best)   return "\\textbf{" + cell + "}";
        if (rowIdx == rp.second) return "\\underline{" + cell + "}";
        return cell;
    };

    std::ostringstream oss;
    oss << "\\begin{table}[htbp]\n"
        << "\\centering\n"
        << "\\caption{Dataset Quantitative Comparison (Mean $\\pm$ Std)}\n"
        << "\\begin{tabular}{lcccccc}\n"
        << "\\toprule\n"
        << "Method & PSNR (dB) $\\uparrow$ & IE $\\uparrow$ & AG $\\uparrow$ & PIQE $\\downarrow$ & Time (ms) $\\downarrow$ & Win Rate (\\%) $\\uparrow$ \\\\\n"
        << "\\midrule\n";

    for (int i = 0; i < static_cast<int>(summaries.size()); ++i) {
        const auto& s = summaries[static_cast<std::size_t>(i)];
        const bool isOrig = (s.algoId == BenchmarkAlgorithmId::Original);
        oss << s.algoName << " & "
            << (isOrig ? "-" : decorate(detail::fmtMeanStdLatex(s.meanPsnr, s.stdPsnr), i, rPsnr)) << " & "
            << decorate(detail::fmtMeanStdLatex(s.meanIE, s.stdIE), i, rIE) << " & "
            << decorate(detail::fmtMeanStdLatex(s.meanAG, s.stdAG), i, rAG) << " & "
            << decorate(detail::fmtMeanStdLatex(s.meanPIQE, s.stdPIQE), i, rPiqe) << " & "
            << (isOrig ? "-" : decorate(detail::fmtMeanStdLatex(s.meanMs, s.stdMs), i, rMs)) << " & "
            << (isOrig ? "-" : decorate(detail::fmtNum(s.winRatePct, 1) + "\\%", i, rWin))
            << " \\\\\n";
    }

    oss << "\\bottomrule\n"
        << "\\end{tabular}\n"
        << "\\end{table}\n";
    return oss.str();
}

/**
 * @brief Generate a GitHub-flavored Markdown summary table with **bold** best values.
 */
inline std::string exportSummaryToMarkdown(const std::vector<AlgoSummaryRow>& summaries)
{
    const auto rPsnr = detail::findBestAndSecond(summaries, [](const auto& s) { return s.meanPsnr; }, true);
    const auto rIE   = detail::findBestAndSecond(summaries, [](const auto& s) { return s.meanIE; }, true);
    const auto rAG   = detail::findBestAndSecond(summaries, [](const auto& s) { return s.meanAG; }, true);
    const auto rPiqe = detail::findBestAndSecond(summaries, [](const auto& s) { return s.meanPIQE; }, false);
    const auto rMs   = detail::findBestAndSecond(summaries, [](const auto& s) { return s.meanMs; }, false);
    const auto rWin  = detail::findBestAndSecond(summaries, [](const auto& s) { return s.winRatePct; }, true);

    auto decorate = [](const std::string& cell, int rowIdx, const detail::RankPair& rp) -> std::string {
        if (rowIdx == rp.best) return "**" + cell + "**";
        return cell;
    };

    std::ostringstream oss;
    oss << "| Method | PSNR (dB) ↑ | IE ↑ | AG ↑ | PIQE ↓ | Time (ms) ↓ | Win Rate (%) ↑ |\n"
        << "|---|---:|---:|---:|---:|---:|---:|\n";

    for (int i = 0; i < static_cast<int>(summaries.size()); ++i) {
        const auto& s = summaries[static_cast<std::size_t>(i)];
        const bool isOrig = (s.algoId == BenchmarkAlgorithmId::Original);
        oss << "| " << s.algoName << " | "
            << (isOrig ? "—" : decorate(detail::fmtMeanStdText(s.meanPsnr, s.stdPsnr), i, rPsnr)) << " | "
            << decorate(detail::fmtMeanStdText(s.meanIE, s.stdIE), i, rIE) << " | "
            << decorate(detail::fmtMeanStdText(s.meanAG, s.stdAG), i, rAG) << " | "
            << decorate(detail::fmtMeanStdText(s.meanPIQE, s.stdPIQE), i, rPiqe) << " | "
            << (isOrig ? "—" : decorate(detail::fmtMeanStdText(s.meanMs, s.stdMs), i, rMs)) << " | "
            << (isOrig ? "—" : decorate(detail::fmtNum(s.winRatePct, 1) + "%", i, rWin))
            << " |\n";
    }
    return oss.str();
}

} // namespace msgf
