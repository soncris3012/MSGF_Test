#include "core/DatasetManager.h"
#include "core/MsgfAlgorithm.h"
#include "core/MsgfMetrics.h"
#include "core/MsgfParams.h"
#include "core/N5Algorithm.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void printUsage(const char* prog)
{
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --input <path>       Input image file or directory (default: sample_images)\n"
              << "  --algo <msgf|n5>     Enhancement algorithm (default: msgf)\n"
              << "  --output-dir <dir>   Save enhanced PNGs (and optional layers) to directory\n"
              << "  --save-layers        Also export B3, D1..D3, IB_clahe, ID_fused per image\n"
              << "  --csv <file>         Write CSV metrics report\n"
              << "  --max <N>            Limit to N evenly-strided images (0 = all)\n"
              << "  --all                Include *_paper_ours and fig3_* files when scanning dir\n"
              << "  --r1 <int>           Scale 1 radius (default 2)\n"
              << "  --r2 <int>           Scale 2 radius (default 6)\n"
              << "  --r3 <int>           Scale 3 radius (default 16)\n"
              << "  --eps1 <float>       Scale 1 eps on [0,1] (default 0.005)\n"
              << "  --eps2 <float>       Scale 2 eps on [0,1] (default 0.02)\n"
              << "  --eps3 <float>       Scale 3 eps on [0,1] (default 0.08)\n"
              << "  --clip <float>       CLAHE clipLimit c (default 2.0)\n"
              << "  --grid <int>         CLAHE tileGrid n (default 8)\n"
              << "  --alpha1 <float>     Dynamic linear alpha1 (default 1.5)\n"
              << "  --alpha2 <float>     Dynamic linear alpha2 (default 1.5)\n"
              << "  --alpha3 <float>     Dynamic linear alpha3 (default 1.5)\n"
              << "  --w1 <float>         Detail weight w1 (default 1.0)\n"
              << "  --w2 <float>         Detail weight w2 (default 1.0)\n"
              << "  --w3 <float>         Detail weight w3 (default 1.0)\n"
              << "  --p <float>          Base layer weight p (default 1.0)\n"
              << "  --q <float>          Detail layer weight q (default 1.0)\n"
              << "  --help               Show this help\n";
}

bool isImageExt(const fs::path& path)
{
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg"
        || ext == ".bmp" || ext == ".tif" || ext == ".tiff";
}

} // namespace

int main(int argc, char* argv[])
{
    fs::path inputPath = "sample_images";
    fs::path outputDir;
    fs::path csvPath;
    std::string algoMode = "msgf";
    bool saveLayers = false;
    bool includeAll = false;
    int maxImages = 0;
    MsgfParams params;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto needVal = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << '\n';
                std::exit(2);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--algo") {
            algoMode = needVal("--algo");
        } else if (arg == "--input") {
            inputPath = needVal("--input");
        } else if (arg == "--output-dir") {
            outputDir = needVal("--output-dir");
        } else if (arg == "--csv") {
            csvPath = needVal("--csv");
        } else if (arg == "--save-layers") {
            saveLayers = true;
        } else if (arg == "--max") {
            maxImages = std::stoi(needVal("--max"));
        } else if (arg == "--all") {
            includeAll = true;
        } else if (arg == "--r1") {
            params.r1 = std::stoi(needVal("--r1"));
        } else if (arg == "--r2") {
            params.r2 = std::stoi(needVal("--r2"));
        } else if (arg == "--r3") {
            params.r3 = std::stoi(needVal("--r3"));
        } else if (arg == "--eps1") {
            params.eps1 = std::stod(needVal("--eps1"));
        } else if (arg == "--eps2") {
            params.eps2 = std::stod(needVal("--eps2"));
        } else if (arg == "--eps3") {
            params.eps3 = std::stod(needVal("--eps3"));
        } else if (arg == "--clip") {
            params.claheClipLimit = std::stod(needVal("--clip"));
        } else if (arg == "--grid") {
            params.claheTileGrid = std::stoi(needVal("--grid"));
        } else if (arg == "--alpha1") {
            params.alpha1 = std::stod(needVal("--alpha1"));
        } else if (arg == "--alpha2") {
            params.alpha2 = std::stod(needVal("--alpha2"));
        } else if (arg == "--alpha3") {
            params.alpha3 = std::stod(needVal("--alpha3"));
        } else if (arg == "--w1") {
            params.w1 = std::stod(needVal("--w1"));
        } else if (arg == "--w2") {
            params.w2 = std::stod(needVal("--w2"));
        } else if (arg == "--w3") {
            params.w3 = std::stod(needVal("--w3"));
        } else if (arg == "--p") {
            params.p = std::stod(needVal("--p"));
        } else if (arg == "--q") {
            params.q = std::stod(needVal("--q"));
        } else {
            std::cerr << "Unknown argument: " << arg << '\n';
            return 2;
        }
    }

    std::vector<fs::path> files;
    std::error_code ec;
    if (fs::is_regular_file(inputPath, ec)) {
        files.push_back(inputPath);
    } else if (fs::is_directory(inputPath, ec)) {
        if (includeAll) {
            for (const auto& entry : fs::directory_iterator(inputPath, ec)) {
                if (entry.is_regular_file(ec) && isImageExt(entry.path())) {
                    files.push_back(entry.path());
                }
            }
            std::sort(files.begin(), files.end());
            if (maxImages > 0 && static_cast<int>(files.size()) > maxImages) {
                files.resize(static_cast<std::size_t>(maxImages));
            }
        } else {
            const auto scanned = msgf::DatasetManager::scanThermalImages(inputPath.string(), maxImages);
            files.reserve(scanned.size());
            for (const std::string& p : scanned) {
                files.emplace_back(p);
            }
        }
    } else {
        std::cerr << "Input path does not exist: " << inputPath << '\n';
        return 1;
    }

    if (files.empty()) {
        std::cerr << "No input images found in: " << inputPath << '\n';
        return 1;
    }

    if (!outputDir.empty()) {
        fs::create_directories(outputDir, ec);
    }

    std::ofstream csv;
    if (!csvPath.empty()) {
        if (csvPath.has_parent_path()) {
            fs::create_directories(csvPath.parent_path(), ec);
        }
        csv.open(csvPath);
        csv << "file,width,height,time_ms,psnr_db,ie_in,ie_out,ag_in,ag_out,piqe_in,piqe_out,mae\n";
    }

    std::cout << std::left << std::setw(26) << "File"
              << std::right << std::setw(10) << "Time(ms)"
              << std::setw(10) << "PSNR(dB)"
              << std::setw(15) << "IE (in->out)"
              << std::setw(17) << "AG (in->out)"
              << std::setw(19) << "PIQE (in->out)" << '\n';
    std::cout << std::string(97, '-') << '\n';

    double sumMs = 0.0;
    double sumPsnr = 0.0;
    double sumIeOut = 0.0;
    double sumAgOut = 0.0;
    double sumPiqeOut = 0.0;
    int processed = 0;

    for (const auto& file : files) {
        const cv::Mat img = cv::imread(file.string(), cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            std::cerr << "Warning: failed to read " << file << '\n';
            continue;
        }

        cv::Mat in8U;
        cv::Mat out8U;
        double elapsedMs = 0.0;
        MsgfOutputs msgfOut;
        msgf::N5Outputs n5Out;

        if (algoMode == "n5") {
            n5Out = msgf::N5Algorithm::run(img, msgf::N5Params{});
            in8U = n5Out.input8U;
            out8U = n5Out.output8U;
            elapsedMs = n5Out.elapsedMs;
        } else {
            msgfOut = MsgfAlgorithm::run(img, params);
            in8U = msgfOut.input8U;
            out8U = msgfOut.output8U;
            elapsedMs = msgfOut.elapsedMs;
        }
        const MsgfMetricResult m = MsgfMetrics::evaluate(in8U, out8U);

        sumMs += elapsedMs;
        sumPsnr += m.psnr;
        sumIeOut += m.ie;
        sumAgOut += m.ag;
        sumPiqeOut += m.piqe;
        processed++;

        std::ostringstream ieStr, agStr, piqeStr;
        ieStr << std::fixed << std::setprecision(2) << m.inputStats.ie << "->" << m.ie;
        agStr << std::fixed << std::setprecision(2) << m.inputStats.ag << "->" << m.ag;
        piqeStr << std::fixed << std::setprecision(2) << m.inputStats.piqe << "->" << m.piqe;

        std::cout << std::left << std::setw(26) << file.filename().string()
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(10) << elapsedMs
                  << std::setw(10) << m.psnr
                  << std::setw(15) << ieStr.str()
                  << std::setw(17) << agStr.str()
                  << std::setw(19) << piqeStr.str() << '\n';

        if (csv.is_open()) {
            csv << file.filename().string() << ','
                << out8U.cols << ',' << out8U.rows << ','
                << std::fixed << std::setprecision(3) << elapsedMs << ','
                << m.psnr << ','
                << m.inputStats.ie << ',' << m.ie << ','
                << m.inputStats.ag << ',' << m.ag << ','
                << m.inputStats.piqe << ',' << m.piqe << ','
                << m.mae << '\n';
        }

        if (!outputDir.empty()) {
            const std::string stem = file.stem().string();
            const std::string suffix = (algoMode == "n5") ? "_n5.png" : "_msgf.png";
            cv::imwrite((outputDir / (stem + suffix)).string(), out8U);
            if (saveLayers && algoMode != "n5") {
                auto save01 = [&](const std::string& layerSuffix, const cv::Mat& layer) {
                    cv::Mat clamped, u8;
                    cv::min(cv::max(layer, 0.0f), 1.0f, clamped);
                    clamped.convertTo(u8, CV_8U, 255.0);
                    cv::imwrite((outputDir / (stem + layerSuffix)).string(), u8);
                };
                save01("_B3.png", msgfOut.B3);
                save01("_IB_clahe.png", msgfOut.IB_clahe);
                cv::imwrite((outputDir / (stem + "_D1.png")).string(), MsgfAlgorithm::normalizeForDisplay(msgfOut.D1));
                cv::imwrite((outputDir / (stem + "_D2.png")).string(), MsgfAlgorithm::normalizeForDisplay(msgfOut.D2));
                cv::imwrite((outputDir / (stem + "_D3.png")).string(), MsgfAlgorithm::normalizeForDisplay(msgfOut.D3));
                cv::imwrite((outputDir / (stem + "_ID_fused.png")).string(), MsgfAlgorithm::normalizeForDisplay(msgfOut.ID_fused));
            }
        }
    }

    if (processed > 0) {
        std::cout << std::string(97, '-') << '\n';
        std::cout << std::left << std::setw(26) << ("Mean (" + std::to_string(processed) + " images)")
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(10) << (sumMs / processed)
                  << std::setw(10) << (sumPsnr / processed)
                  << std::setw(15) << (sumIeOut / processed)
                  << std::setw(17) << (sumAgOut / processed)
                  << std::setw(19) << (sumPiqeOut / processed) << '\n';
    }

    std::cout << "\nPaper Table 1 Reference (MSGF-TIR Ours — Li et al., Fire 2024, 7, 192):\n";
    for (const auto& row : msgf::paperTable1Benchmarks()) {
        if (!row.isProposed) continue;
        std::cout << "  " << std::left << std::setw(8) << row.sceneName
                  << " (" << std::setw(11) << row.datasetSource << ") "
                  << std::fixed << std::setprecision(2)
                  << "PSNR=" << std::setw(5) << row.psnr
                  << "  IE=" << std::setw(4) << row.ie
                  << "  AG=" << std::setw(5) << row.ag
                  << "  PIQE=" << std::setw(5) << row.piqe << '\n';
    }

    return 0;
}
