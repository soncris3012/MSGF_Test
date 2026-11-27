#pragma once

#include "core/BenchmarkTypes.h"
#include "core/DatasetManager.h"
#include "core/MsgfParams.h"
#include "core/N5Algorithm.h"

#include <QFutureWatcher>
#include <QString>
#include <QStringList>
#include <QWidget>

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QSpinBox;
class QTableWidget;
class QChartView;

/**
 * @brief Interactive Dataset Benchmark & Charts tab for evaluating MSGF-TIR
 *        against baseline methods (HE, GF&DDE, BF&DDE) across entire folders.
 */
class DatasetBenchmarkWidget : public QWidget {
    Q_OBJECT

public:
    explicit DatasetBenchmarkWidget(QWidget* parent = nullptr);
    ~DatasetBenchmarkWidget() override;

    void setAvailableDatasets(
        const std::vector<msgf::DatasetEntry>& datasets,
        const QString& defaultRoot = {}
    );
    void setDatasetDirectory(const QString& dirPath);
    void setMsgfParams(const MsgfParams& params);
    void setN5Params(const msgf::N5Params& params);
    void setDefaultFocusAlgorithm(msgf::BenchmarkAlgorithmId id);
    [[nodiscard]] bool isRunning() const noexcept;

signals:
    void imageActivated(const QString& filePath);

private slots:
    void onChooseDataset();
    void onBenchDatasetSelected(int index);
    void onMaxImagesChanged(int value);
    void onRunBenchmark();
    void onCancelBenchmark();
    void onExportCsv();
    void onCopyLatex();
    void onCopyMarkdown();
    void onSaveChartPng();
    void onChartConfigChanged();
    void onFilterChanged();
    void onDetailAlgoChanged(int index);
    void onDetailCellDoubleClicked(int row, int column);
    void onBenchmarkFinished();

private:
    struct PerImageBatchResult {
        std::map<msgf::BenchmarkAlgorithmId, msgf::ImageBenchmarkRow> rowsByAlgo;
    };

    void setupUi();
    void setupConnections();
    void scanDirectory(const QString& dirPath);
    void setRunningState(bool running);
    void refreshSummaryTable();
    void rebuildDetailTable();
    void appendDetailRow(const msgf::ImageBenchmarkRow& row);
    void applyDetailFilters();
    void refreshChart();

    static void highlightBestAndSecondInColumn(
        QTableWidget* table,
        int col,
        bool higherIsBetter
    );

    [[nodiscard]] std::vector<msgf::BenchmarkAlgorithmId> selectedAlgorithms() const;

    QString m_defaultRoot;
    QString m_datasetDir;
    QStringList m_imageFiles;
    MsgfParams m_msgfParams;
    msgf::N5Params m_n5Params;
    msgf::BenchmarkAlgorithmId m_focusAlgo = msgf::BenchmarkAlgorithmId::MSGF_TIR;

    std::vector<msgf::BenchmarkAlgorithmId> m_lastOrderedAlgos;
    std::map<msgf::BenchmarkAlgorithmId, std::vector<msgf::ImageBenchmarkRow>> m_perAlgoRows;
    std::vector<msgf::AlgoSummaryRow> m_summaries;

    std::uint64_t m_runId = 0;
    bool m_isRunning = false;
    std::shared_ptr<std::atomic<bool>> m_cancelFlag;
    QFutureWatcher<void>* m_watcher = nullptr;

    // Controls
    QComboBox* m_comboBenchDataset = nullptr;
    QPushButton* m_btnChooseDataset = nullptr;
    QSpinBox* m_spinMaxImages = nullptr;
    QLabel* m_lblDatasetPath = nullptr;
    QCheckBox* m_chkHE = nullptr;
    QCheckBox* m_chkGFDDE = nullptr;
    QCheckBox* m_chkBFDDE = nullptr;
    QCheckBox* m_chkMSGF = nullptr;
    QCheckBox* m_chkN5 = nullptr;

    QPushButton* m_btnRun = nullptr;
    QPushButton* m_btnCancel = nullptr;
    QProgressBar* m_progressBar = nullptr;
    QLabel* m_lblStatus = nullptr;
    QPushButton* m_btnExportCsv = nullptr;
    QPushButton* m_btnCopyLatex = nullptr;
    QPushButton* m_btnCopyMarkdown = nullptr;

    // Summary Table & Charts
    QTableWidget* m_tblSummary = nullptr;
    QComboBox* m_comboChartType = nullptr;
    QComboBox* m_comboChartMetric = nullptr;
    QPushButton* m_btnSaveChart = nullptr;
    QChartView* m_chartView = nullptr;

    // Detail Table & Filters
    QLineEdit* m_editSearch = nullptr;
    QComboBox* m_comboFilter = nullptr;
    QComboBox* m_comboDetailAlgo = nullptr;
    QLabel* m_lblDetailInfo = nullptr;
    QTableWidget* m_tblDetail = nullptr;
};
