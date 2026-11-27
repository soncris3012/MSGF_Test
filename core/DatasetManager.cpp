#include "core/DatasetManager.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <unordered_set>

namespace fs = std::filesystem;

namespace msgf {
namespace {

std::string toLowerCopy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

bool shouldSkipSubdir(const std::string& lowerDirName)
{
    if (lowerDirName.empty() || lowerDirName[0] == '.') {
        return true;
    }
    static const std::unordered_set<std::string> kSkipNames = {
        "visible",
        "vi",
        "images_rgb_train",
        "images_rgb_val",
        "video_rgb_test",
        "crop_hr_visible",
        "crop_lr_visible",
        "segmentation_labels",
        "labels",
        "analyticsdata",
        "archives",
        "manifests",
        "splits"
    };
    return kSkipNames.find(lowerDirName) != kSkipNames.end();
}

bool isBuiltInSampleHelperFile(const fs::path& dirPath, const std::string& fileName)
{
    const std::string dirName = toLowerCopy(dirPath.filename().string());
    if (dirName != "sample_images") {
        return false;
    }
    const std::string lower = toLowerCopy(fileName);
    return lower.find("_paper_ours") != std::string::npos
        || lower.rfind("fig3_", 0) == 0;
}

std::vector<std::string> applyEvenStrideLimit(std::vector<std::string> files, int maxLimit)
{
    if (maxLimit <= 0 || static_cast<int>(files.size()) <= maxLimit) {
        return files;
    }
    std::vector<std::string> sampled;
    sampled.reserve(static_cast<std::size_t>(maxLimit));
    const double step = static_cast<double>(files.size()) / static_cast<double>(maxLimit);
    for (int i = 0; i < maxLimit; ++i) {
        const std::size_t idx = std::min<std::size_t>(
            files.size() - 1,
            static_cast<std::size_t>(std::floor(static_cast<double>(i) * step))
        );
        sampled.push_back(files[idx]);
    }
    return sampled;
}

int countDirectImagesFast(const fs::path& dir, bool filterSampleHelpers)
{
    std::error_code ec;
    int count = 0;
    for (const auto& entry : fs::directory_iterator(
             dir, fs::directory_options::skip_permission_denied, ec)) {
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        const std::string fname = entry.path().filename().string();
        if (!DatasetManager::isImageExtension(fname)) {
            continue;
        }
        if (filterSampleHelpers && isBuiltInSampleHelperFile(dir, fname)) {
            continue;
        }
        count++;
    }
    return count;
}

} // namespace

bool DatasetManager::isImageExtension(const std::string& filenameOrExt)
{
    const fs::path p(filenameOrExt);
    std::string ext = p.has_extension() ? p.extension().string() : filenameOrExt;
    ext = toLowerCopy(std::move(ext));
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg"
        || ext == ".bmp" || ext == ".tif" || ext == ".tiff";
}

std::string DatasetManager::detectDatasetRoot(const std::string& overrideRoot)
{
    std::error_code ec;
    if (!overrideRoot.empty() && fs::exists(overrideRoot, ec) && fs::is_directory(overrideRoot, ec)) {
        return fs::canonical(overrideRoot, ec).string();
    }

    for (const char* envName : {"IR_DATASET_DIR", "QART_DATASETS_DIR"}) {
        if (const char* val = std::getenv(envName)) {
            if (*val && fs::exists(val, ec) && fs::is_directory(val, ec)) {
                return fs::canonical(val, ec).string();
            }
        }
    }

    std::string home;
#if defined(_WIN32)
    if (const char* userProfile = std::getenv("USERPROFILE")) {
        home = userProfile;
    }
#else
    if (const char* homeEnv = std::getenv("HOME")) {
        home = homeEnv;
    }
#endif

    if (!home.empty()) {
        const std::string homeCandidates[] = {
            home + "/QART_Datasets/raw",
            home + "/QART_Datasets",
            home + "/datasets/raw",
            home + "/datasets"
        };
        for (const auto& c : homeCandidates) {
            if (fs::exists(c, ec) && fs::is_directory(c, ec)) {
                return fs::canonical(c, ec).string();
            }
        }
    }

    const std::string relCandidates[] = {
        "./QART_Datasets/raw",
        "../QART_Datasets/raw",
        "../../QART_Datasets/raw",
        "./datasets/raw",
        "./datasets"
    };
    for (const auto& rel : relCandidates) {
        if (fs::exists(rel, ec) && fs::is_directory(rel, ec)) {
            return fs::canonical(rel, ec).string();
        }
    }

    return "";
}

std::vector<std::string> DatasetManager::scanThermalImages(const std::string& dirPath, int maxLimit)
{
    std::vector<std::string> files;
    std::error_code ec;
    if (dirPath.empty() || !fs::exists(dirPath, ec) || !fs::is_directory(dirPath, ec)) {
        return files;
    }

    fs::path root(dirPath);
    const bool filterSample = (toLowerCopy(root.filename().string()) == "sample_images");

    // 1. Check direct image files inside dirPath first
    for (const auto& entry : fs::directory_iterator(
             root, fs::directory_options::skip_permission_denied, ec)) {
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        const std::string fname = entry.path().filename().string();
        if (!isImageExtension(fname)) {
            continue;
        }
        if (filterSample && isBuiltInSampleHelperFile(root, fname)) {
            continue;
        }
        files.push_back(entry.path().string());
    }

    // 2. If user selected a parent directory (e.g. QART_Datasets, raw, llvip, llvip/infrared, flir, msrs, roadscene),
    //    recursively collect thermal images while skipping RGB/visible/label/archive folders.
    if (files.empty()) {
        if (fs::exists(root / "raw", ec) && fs::is_directory(root / "raw", ec)) {
            root = root / "raw";
        }

        fs::recursive_directory_iterator it(
            root,
            fs::directory_options::skip_permission_denied,
            ec
        );
        const fs::recursive_directory_iterator end;
        while (it != end) {
            if (ec) {
                ec.clear();
                ++it;
                continue;
            }
            const auto& entry = *it;
            if (entry.is_directory(ec)) {
                const std::string subName = toLowerCopy(entry.path().filename().string());
                if (shouldSkipSubdir(subName)) {
                    it.disable_recursion_pending();
                }
            } else if (entry.is_regular_file(ec)) {
                const std::string fname = entry.path().filename().string();
                if (isImageExtension(fname) && !isBuiltInSampleHelperFile(entry.path().parent_path(), fname)) {
                    files.push_back(entry.path().string());
                }
            }
            it.increment(ec);
        }
    }

    std::sort(files.begin(), files.end());
    return applyEvenStrideLimit(std::move(files), maxLimit);
}

std::vector<DatasetEntry> DatasetManager::discoverDatasets(
    const std::string& datasetRoot,
    const std::string& sampleImagesDir)
{
    std::vector<DatasetEntry> result;
    std::error_code ec;

    // 1. Built-in paper sample scenes
    if (!sampleImagesDir.empty() && fs::exists(sampleImagesDir, ec) && fs::is_directory(sampleImagesDir, ec)) {
        const fs::path sp(sampleImagesDir);
        const int count = countDirectImagesFast(sp, true);
        if (count > 0) {
            result.push_back({
                "paper_samples",
                "Paper Samples (Fire 2024 - " + std::to_string(count) + " imgs)",
                fs::canonical(sp, ec).string(),
                count
            });
        }
    }

    if (datasetRoot.empty() || !fs::exists(datasetRoot, ec) || !fs::is_directory(datasetRoot, ec)) {
        return result;
    }

    fs::path root(datasetRoot);
    if (fs::exists(root / "raw", ec) && fs::is_directory(root / "raw", ec)) {
        root = root / "raw";
    }

    struct Candidate {
        const char* id;
        const char* subPath;
        const char* prettyName;
    };

    const Candidate candidates[] = {
        {"flir_val",        "flir/images_thermal_val/data",   "FLIR (Thermal Val)"},
        {"flir_train",      "flir/images_thermal_train/data", "FLIR (Thermal Train)"},
        {"flir_video",      "flir/video_thermal_test/data",   "FLIR (Video Thermal Test)"},
        {"llvip_test",      "llvip/infrared/test",            "LLVIP (Test IR)"},
        {"llvip_train",     "llvip/infrared/train",           "LLVIP (Train IR)"},
        {"msrs_test",       "msrs/test/ir",                   "MSRS (Test IR)"},
        {"msrs_train",      "msrs/train/ir",                  "MSRS (Train IR)"},
        {"msrs_det",        "msrs/detection/ir",              "MSRS (Detection IR)"},
        {"roadscene_ir",    "roadscene/infrared",             "RoadScene (IR)"},
        {"roadscene_crop",  "roadscene/cropinfrared",         "RoadScene (Crop IR)"}
    };

    for (const auto& c : candidates) {
        const fs::path p = root / c.subPath;
        if (!fs::exists(p, ec) || !fs::is_directory(p, ec)) {
            continue;
        }
        const int count = countDirectImagesFast(p, false);
        if (count > 0) {
            result.push_back({
                c.id,
                std::string(c.prettyName) + " (" + std::to_string(count) + " imgs)",
                fs::canonical(p, ec).string(),
                count
            });
        }
    }

    return result;
}

} // namespace msgf
