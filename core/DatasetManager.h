#pragma once

#include <string>
#include <vector>

namespace msgf {

struct DatasetEntry {
    std::string id;
    std::string displayName;
    std::string folderPath;
    int imageCount = 0;
};

/**
 * @brief Cross-platform dataset auto-detection and recursive thermal image scanner
 *        supporting QART_Datasets (FLIR, LLVIP, MSRS, RoadScene) as well as
 *        arbitrary parent/leaf thermal folders.
 */
class DatasetManager {
public:
    /**
     * @brief Auto-detect root dataset folder:
     *        1. overrideRoot if valid directory
     *        2. Env vars: IR_DATASET_DIR, QART_DATASETS_DIR
     *        3. Standard home paths: ~/QART_Datasets/raw, ~/QART_Datasets, ~/datasets/raw, ~/datasets
     *        4. Workspace relative paths
     */
    static std::string detectDatasetRoot(const std::string& overrideRoot = "");

    /**
     * @brief Discover built-in Paper Samples plus QART_Datasets thermal subsets
     *        (FLIR, LLVIP, MSRS, RoadScene).
     */
    static std::vector<DatasetEntry> discoverDatasets(
        const std::string& datasetRoot,
        const std::string& sampleImagesDir = ""
    );

    /**
     * @brief Collect thermal images from dirPath:
     *        - Direct image files in dirPath if present (filtering sample_images refs)
     *        - Otherwise recursive scan skipping RGB/visible/label/archive folders
     *        - Optional evenly-strided sampling if maxLimit > 0
     */
    static std::vector<std::string> scanThermalImages(
        const std::string& dirPath,
        int maxLimit = 0
    );

    static bool isImageExtension(const std::string& filenameOrExt);
};

} // namespace msgf
