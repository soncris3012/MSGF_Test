#pragma once

#include "core/DatasetManager.h"
#include "core/MsgfAlgorithm.h"
#include "core/MsgfMetrics.h"
#include "core/MsgfParams.h"

#include <QFutureWatcher>
#include <QImage>
#include <QKeyEvent>
#include <QPointF>
#include <QString>
#include <QTimer>
#include <QWidget>

#include <cstdint>
#include <memory>
#include <vector>

namespace Ui {
class MsgfForm;
}

/**
 * @brief Self-contained QWidget form implementing the MSGF-TIR interactive bench.
 *        Can be run standalone (via main.cpp) or embedded directly inside any
 *        QTabWidget / QMainWindow.
 */
class MsgfForm : public QWidget {
    Q_OBJECT

public:
    explicit MsgfForm(QWidget* parent = nullptr);
    ~MsgfForm() override;

    /**
     * @brief Programmatically inject an image (useful when embedded in a host window).
     */
    void setInputImage(const cv::Mat& image, const QString& label = QStringLiteral("External Input"));
    bool loadImageFile(const QString& filePath);

    [[nodiscard]] MsgfParams currentParams() const;
    void setParams(const MsgfParams& params);

signals:
    void processingFinished(const msgf::MsgfOutputs& outputs, const msgf::MsgfMetricResult& metrics);

protected:
    void keyPressEvent(QKeyEvent* event) override;

private slots:
    void onLoadImage();
    void onOpenFolder();
    void onSetDatasetRoot();
    void onDatasetSelected(int index);
    void onSceneSelected(int index);
    void onPrevImage();
    void onNextImage();
    void onParamEdited();
    void onRunClicked();
    void onResetDefaults();
    void onSaveOutput();
    void onExportStages();
    void onWorkerFinished();
    void onSyncViewChanged(double scale, const QPointF& offset);
    void onSyncViewReset();

private:
    struct WorkerResult {
        std::uint64_t generation = 0;
        bool ok = false;
        QString errorText;
        MsgfOutputs outputs;
        MsgfMetricResult metrics;
    };

    void setupConnections();
    void applyTheme();
    void populateBenchmarkTable();
    void initializeSampleDirectory();
    void refreshDatasets(const QString& rootOverride = {});
    void loadFolderImages(const QString& dirPath);
    void updateNavigationButtons();
    void scheduleProcessing(bool immediate = false);
    void startBackgroundRun();
    void updateViewers();
    void updateMetricsDisplay();
    [[nodiscard]] std::vector<class ImageCanvas*> allCanvases() const;

    static QString locateSampleDirectory();
    static QImage matToQImage(const cv::Mat& mat);

    std::unique_ptr<Ui::MsgfForm> ui;

    QString m_sampleDir;
    QString m_datasetRoot;
    std::vector<msgf::DatasetEntry> m_datasets;
    std::vector<QString> m_imageFiles;
    int m_currentFileIndex = -1;
    QString m_currentLabel;

    cv::Mat m_inputMat;
    cv::Mat m_paperRefMat;
    MsgfOutputs m_lastOutputs;
    MsgfMetricResult m_lastMetrics;

    bool m_isUpdatingControls = false;
    bool m_isSyncingViews = false;
    std::uint64_t m_generation = 0;

    QTimer* m_debounceTimer = nullptr;
    QFutureWatcher<WorkerResult>* m_watcher = nullptr;
};
