#pragma once

#include "core/DatasetManager.h"
#include "core/MsgfMetrics.h"
#include "core/N5Algorithm.h"

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
class N5Form;
}

class ImageCanvas;

/**
 * @brief Self-contained QWidget form implementing the Accepted N5 interactive
 *        workbench & stage-by-stage ablation viewer.
 */
class N5Form : public QWidget {
    Q_OBJECT

public:
    explicit N5Form(QWidget* parent = nullptr);
    ~N5Form() override;

    void setInputImage(const cv::Mat& image, const QString& label = QStringLiteral("External Input"));
    bool loadImageFile(const QString& filePath);

    [[nodiscard]] msgf::N5Params currentParams() const;
    void setParams(const msgf::N5Params& params);

signals:
    void processingFinished(const msgf::N5Outputs& outputs, const msgf::MsgfMetricResult& metrics);

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
    void onFreezeToggled(bool checked);
    void onParamEdited();
    void onRunClicked();
    void onResetAccepted();
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
        msgf::N5Outputs outputs;
        msgf::MsgfMetricResult finalMetrics;
        msgf::MsgfMetricResult claheMetrics;
        msgf::MsgfMetricResult stage1Metrics;
    };

    void setupConnections();
    void applyTheme();
    void setupAblationTable();
    void initializeDatasets();
    void refreshDatasets(const QString& rootOverride = {});
    void loadFolderImages(const QString& dirPath);
    void updateNavigationButtons();
    void updateFreezeControls();
    void scheduleProcessing(bool immediate = false);
    void startBackgroundRun();
    void updateViewers();
    void updateMetricsDisplay(const WorkerResult& res);
    [[nodiscard]] std::vector<ImageCanvas*> allCanvases() const;

    static QString locateSampleDirectory();
    static QImage matToQImage(const cv::Mat& mat);

    std::unique_ptr<Ui::N5Form> ui;

    QString m_sampleDir;
    QString m_datasetRoot;
    std::vector<msgf::DatasetEntry> m_datasets;
    std::vector<QString> m_imageFiles;
    int m_currentFileIndex = -1;
    QString m_currentLabel;

    cv::Mat m_inputMat;
    msgf::N5Outputs m_lastOutputs;
    msgf::MsgfMetricResult m_lastMetrics;

    bool m_isUpdatingControls = false;
    bool m_isSyncingViews = false;
    std::uint64_t m_generation = 0;

    QTimer* m_debounceTimer = nullptr;
    QFutureWatcher<WorkerResult>* m_watcher = nullptr;
};
