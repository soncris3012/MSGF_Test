#include "ui/DatasetBenchmarkWidget.h"

#include "core/BenchmarkExport.h"
#include "core/DatasetManager.h"
#include "core/MsgfAlgorithm.h"
#include "core/MsgfBaselines.h"
#include "core/MsgfMetrics.h"
#include "ui/NumericTableItem.h"

#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QComboBox>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMetaObject>
#include <QPainter>
#include <QPointer>
#include <QProgressBar>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QSplitter>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QtCharts/QBarCategoryAxis>
#include <QtCharts/QBarSeries>
#include <QtCharts/QBarSet>
#include <QtCharts/QChart>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QValueAxis>
#include <QtConcurrent/QtConcurrentRun>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <memory>

namespace {

constexpr int kColSummaryMethod  = 0;
constexpr int kColSummaryPsnr    = 1;
constexpr int kColSummaryIE      = 2;
constexpr int kColSummaryAG      = 3;
constexpr int kColSummaryPiqe    = 4;
constexpr int kColSummaryTime    = 5;
constexpr int kColSummaryWinRate = 6;

constexpr int kColDetIdx      = 0;
constexpr int kColDetName     = 1;
constexpr int kColDetSize     = 2;
constexpr int kColDetPsnr     = 3;
constexpr int kColDetOrigIE   = 4;
constexpr int kColDetOutIE    = 5;
constexpr int kColDetDeltaIE  = 6;
constexpr int kColDetOrigAG   = 7;
constexpr int kColDetOutAG    = 8;
constexpr int kColDetDeltaAG  = 9;
constexpr int kColDetOutPiqe  = 10;
constexpr int kColDetDeltaPiqe = 11;
constexpr int kColDetTime     = 12;

QString formatMeanStd(double mean, double stdDev, int prec = 2)
{
    return QStringLiteral("%1 ± %2")
        .arg(mean, 0, 'f', prec)
        .arg(stdDev, 0, 'f', prec);
}

QString formatSignedDelta(double delta, int prec = 2)
{
    const QString sign = (delta >= 0.0) ? QStringLiteral("+") : QStringLiteral("");
    return sign + QString::number(delta, 'f', prec);
}

QColor algoColor(msgf::BenchmarkAlgorithmId id)
{
    switch (id) {
    case msgf::BenchmarkAlgorithmId::Original: return QColor(156, 163, 175);
    case msgf::BenchmarkAlgorithmId::HE:       return QColor(245, 158, 11);
    case msgf::BenchmarkAlgorithmId::GF_DDE:   return QColor(56, 189, 248);
    case msgf::BenchmarkAlgorithmId::BF_DDE:   return QColor(168, 85, 247);
    case msgf::BenchmarkAlgorithmId::MSGF_TIR: return QColor(52, 211, 153);
    }
    return QColor(200, 200, 200);
}

double extractMetricValue(const msgf::AlgoSummaryRow& s, const QString& metricName)
{
    if (metricName == QStringLiteral("PSNR (dB)")) return s.meanPsnr;
    if (metricName == QStringLiteral("IE"))        return s.meanIE;
    if (metricName == QStringLiteral("AG"))        return s.meanAG;
    if (metricName == QStringLiteral("PIQE"))      return s.meanPIQE;
    if (metricName == QStringLiteral("Time (ms)")) return s.meanMs;
    return 0.0;
}

double extractRowMetricValue(const msgf::ImageBenchmarkRow& r, const QString& metricName)
{
    if (metricName == QStringLiteral("PSNR (dB)")) return r.psnr;
    if (metricName == QStringLiteral("IE"))        return r.ie;
    if (metricName == QStringLiteral("AG"))        return r.ag;
    if (metricName == QStringLiteral("PIQE"))      return r.piqe;
    if (metricName == QStringLiteral("Time (ms)")) return r.runtimeMs;
    return 0.0;
}

} // namespace

DatasetBenchmarkWidget::DatasetBenchmarkWidget(QWidget* parent)
    : QWidget(parent)
    , m_cancelFlag(std::make_shared<std::atomic<bool>>(false))
    , m_watcher(new QFutureWatcher<void>(this))
{
    setupUi();
    setupConnections();
    setRunningState(false);
    refreshChart();
}

DatasetBenchmarkWidget::~DatasetBenchmarkWidget()
{
    if (m_cancelFlag) {
        m_cancelFlag->store(true, std::memory_order_relaxed);
    }
    if (m_watcher && m_watcher->isRunning()) {
        m_watcher->waitForFinished();
    }
}

void DatasetBenchmarkWidget::setupUi()
{
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(6, 6, 6, 6);
    root->setSpacing(6);

    // 1. Top Control Panel
    auto* ctrlBox = new QGroupBox(QStringLiteral("Dataset Benchmark Controls"), this);
    auto* ctrlLayout = new QVBoxLayout(ctrlBox);
    ctrlLayout->setSpacing(6);

    // Row 1: Dataset Selector, Folder Picker, Limit & Algorithms
    auto* topRow = new QHBoxLayout();
    m_comboBenchDataset = new QComboBox(ctrlBox);
    m_comboBenchDataset->setObjectName(QStringLiteral("comboBenchDataset"));
    m_comboBenchDataset->setMinimumHeight(28);
    m_comboBenchDataset->setMinimumWidth(210);
    m_comboBenchDataset->setToolTip(QStringLiteral("Chọn nhanh dataset phát hiện tự động (Paper Samples, FLIR, LLVIP, MSRS, RoadScene)"));

    m_btnChooseDataset = new QPushButton(QStringLiteral("Chọn thư mục..."), ctrlBox);
    m_btnChooseDataset->setMinimumHeight(28);

    m_lblDatasetPath = new QLabel(QStringLiteral("Chưa chọn thư mục"), ctrlBox);
    m_lblDatasetPath->setStyleSheet(QStringLiteral("color: #b8bec9;"));
    m_lblDatasetPath->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);

    auto* lblLimit = new QLabel(QStringLiteral("Giới hạn:"), ctrlBox);
    m_spinMaxImages = new QSpinBox(ctrlBox);
    m_spinMaxImages->setObjectName(QStringLiteral("spinMaxImages"));
    m_spinMaxImages->setMinimumHeight(26);
    m_spinMaxImages->setRange(0, 50000);
    m_spinMaxImages->setSingleStep(50);
    m_spinMaxImages->setValue(0);
    m_spinMaxImages->setSpecialValueText(QStringLiteral("Tất cả"));
    m_spinMaxImages->setToolTip(QStringLiteral("0 = chạy toàn bộ ảnh; >0 = lấy mẫu đều N ảnh trên toàn dataset"));

    auto* lblAlgos = new QLabel(QStringLiteral("Thuật toán:"), ctrlBox);
    m_chkHE = new QCheckBox(QStringLiteral("HE"), ctrlBox);
    m_chkHE->setChecked(true);
    m_chkGFDDE = new QCheckBox(QStringLiteral("GF&&DDE"), ctrlBox);
    m_chkGFDDE->setChecked(true);
    m_chkBFDDE = new QCheckBox(QStringLiteral("BF&&DDE"), ctrlBox);
    m_chkBFDDE->setChecked(false);
    m_chkMSGF = new QCheckBox(QStringLiteral("MSGF-TIR"), ctrlBox);
    m_chkMSGF->setChecked(true);
    m_chkMSGF->setStyleSheet(QStringLiteral("font-weight: 600; color: #34d399;"));

    topRow->addWidget(m_comboBenchDataset);
    topRow->addWidget(m_btnChooseDataset);
    topRow->addWidget(m_lblDatasetPath, 1);
    topRow->addWidget(lblLimit);
    topRow->addWidget(m_spinMaxImages);
    topRow->addWidget(lblAlgos);
    topRow->addWidget(m_chkHE);
    topRow->addWidget(m_chkGFDDE);
    topRow->addWidget(m_chkBFDDE);
    topRow->addWidget(m_chkMSGF);
    ctrlLayout->addLayout(topRow);

    // Row 2: Run, Cancel, Progress, Export
    auto* actRow = new QHBoxLayout();
    m_btnRun = new QPushButton(QStringLiteral("▶ Chạy Benchmark"), ctrlBox);
    m_btnRun->setObjectName(QStringLiteral("btnRunBench"));
    m_btnRun->setMinimumHeight(28);
    m_btnRun->setStyleSheet(
        QStringLiteral("background-color: #2857a4; border: 1px solid #3b74d1; font-weight: 600; color: #fff;")
    );

    m_btnCancel = new QPushButton(QStringLiteral("⏹ Hủy"), ctrlBox);
    m_btnCancel->setMinimumHeight(28);

    m_progressBar = new QProgressBar(ctrlBox);
    m_progressBar->setMinimumHeight(22);
    m_progressBar->setRange(0, 1);
    m_progressBar->setValue(0);
    m_progressBar->setFormat(QStringLiteral("%p% (%v/%m ảnh)"));

    m_lblStatus = new QLabel(QStringLiteral("Sẵn sàng"), ctrlBox);
    m_lblStatus->setMinimumWidth(90);

    m_btnExportCsv = new QPushButton(QStringLiteral("Xuất CSV (BOM)"), ctrlBox);
    m_btnExportCsv->setMinimumHeight(28);
    m_btnCopyLatex = new QPushButton(QStringLiteral("Copy LaTeX"), ctrlBox);
    m_btnCopyLatex->setMinimumHeight(28);
    m_btnCopyMarkdown = new QPushButton(QStringLiteral("Copy Markdown"), ctrlBox);
    m_btnCopyMarkdown->setMinimumHeight(28);

    actRow->addWidget(m_btnRun);
    actRow->addWidget(m_btnCancel);
    actRow->addWidget(m_progressBar, 1);
    actRow->addWidget(m_lblStatus);
    actRow->addWidget(m_btnExportCsv);
    actRow->addWidget(m_btnCopyLatex);
    actRow->addWidget(m_btnCopyMarkdown);
    ctrlLayout->addLayout(actRow);

    root->addWidget(ctrlBox);

    // 2. Vertical Splitter (Top: Summary + Charts | Bottom: Detail Table)
    auto* vSplitter = new QSplitter(Qt::Vertical, this);
    vSplitter->setChildrenCollapsible(false);

    auto* hSplitter = new QSplitter(Qt::Horizontal, vSplitter);
    hSplitter->setChildrenCollapsible(false);

    // Left: Summary Table (Mean +/- Std + Win Rate)
    auto* grpSummary = new QGroupBox(QStringLiteral("Bảng 1: Tổng hợp toàn Dataset (Mean ± Std & Win Rate)"), hSplitter);
    auto* sumLayout = new QVBoxLayout(grpSummary);
    sumLayout->setSpacing(4);

    m_tblSummary = new QTableWidget(grpSummary);
    m_tblSummary->setObjectName(QStringLiteral("tblSummary"));
    m_tblSummary->setColumnCount(7);
    m_tblSummary->setHorizontalHeaderLabels({
        QStringLiteral("Method"),
        QStringLiteral("PSNR (dB) ↑"),
        QStringLiteral("IE ↑"),
        QStringLiteral("AG ↑"),
        QStringLiteral("PIQE ↓"),
        QStringLiteral("Time (ms) ↓"),
        QStringLiteral("Win Rate (%) ↑")
    });
    m_tblSummary->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_tblSummary->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_tblSummary->setSelectionMode(QAbstractItemView::SingleSelection);
    m_tblSummary->setAlternatingRowColors(true);
    m_tblSummary->setSortingEnabled(true);
    m_tblSummary->verticalHeader()->setVisible(false);
    m_tblSummary->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_tblSummary->horizontalHeader()->setSectionResizeMode(kColSummaryMethod, QHeaderView::ResizeToContents);

    auto* lblSumHint = new QLabel(
        QStringLiteral("Ghi chú: Xanh lá = Tốt nhất cột | Xanh dương (gạch chân) = Đứng nhì | Bấm tiêu đề cột để sắp xếp theo giá trị số."),
        grpSummary
    );
    lblSumHint->setStyleSheet(QStringLiteral("color: #9aa0aa; font-size: 11px;"));
    lblSumHint->setWordWrap(true);

    sumLayout->addWidget(m_tblSummary, 1);
    sumLayout->addWidget(lblSumHint);

    // Right: Charts Panel (Qt6::Charts)
    auto* grpChart = new QGroupBox(QStringLiteral("Biểu đồ phân tích động (Qt Charts)"), hSplitter);
    auto* chartLayout = new QVBoxLayout(grpChart);
    chartLayout->setSpacing(4);

    auto* chartBar = new QHBoxLayout();
    m_comboChartType = new QComboBox(grpChart);
    m_comboChartType->setObjectName(QStringLiteral("comboChartType"));
    m_comboChartType->addItems({
        QStringLiteral("So sánh cột (Grouped Bar)"),
        QStringLiteral("Xu hướng theo ảnh (Per-Image Trend)"),
        QStringLiteral("Tốc độ vs Chất lượng (Trade-off Scatter)")
    });

    m_comboChartMetric = new QComboBox(grpChart);
    m_comboChartMetric->setObjectName(QStringLiteral("comboChartMetric"));
    m_comboChartMetric->addItems({
        QStringLiteral("PSNR (dB)"),
        QStringLiteral("IE"),
        QStringLiteral("AG"),
        QStringLiteral("PIQE"),
        QStringLiteral("Time (ms)")
    });
    m_comboChartMetric->setCurrentText(QStringLiteral("IE"));

    m_btnSaveChart = new QPushButton(QStringLiteral("Lưu ảnh PNG"), grpChart);
    m_btnSaveChart->setMinimumHeight(26);

    chartBar->addWidget(new QLabel(QStringLiteral("Biểu đồ:"), grpChart));
    chartBar->addWidget(m_comboChartType, 1);
    chartBar->addWidget(new QLabel(QStringLiteral("Chỉ số:"), grpChart));
    chartBar->addWidget(m_comboChartMetric);
    chartBar->addWidget(m_btnSaveChart);
    chartLayout->addLayout(chartBar);

    m_chartView = new QChartView(grpChart);
    m_chartView->setObjectName(QStringLiteral("benchChartView"));
    m_chartView->setRenderHint(QPainter::Antialiasing);
    m_chartView->setMinimumHeight(220);
    chartLayout->addWidget(m_chartView, 1);

    hSplitter->addWidget(grpSummary);
    hSplitter->addWidget(grpChart);
    hSplitter->setSizes({560, 520});

    // Bottom: Detail Table (Per-Image Metrics)
    auto* grpDetail = new QGroupBox(QStringLiteral("Bảng 2: Chi tiết từng ảnh (Double-click dòng để mở ảnh trên Tab Compare & Decomposition)"), vSplitter);
    auto* detLayout = new QVBoxLayout(grpDetail);
    detLayout->setSpacing(4);

    auto* detBar = new QHBoxLayout();
    m_editSearch = new QLineEdit(grpDetail);
    m_editSearch->setPlaceholderText(QStringLiteral("Tìm kiếm theo tên ảnh..."));
    m_editSearch->setClearButtonEnabled(true);

    m_comboFilter = new QComboBox(grpDetail);
    m_comboFilter->addItems({
        QStringLiteral("Tất cả"),
        QStringLiteral("ΔIE > 0 (Tăng thông tin)"),
        QStringLiteral("ΔIE < 0 (Giảm thông tin)"),
        QStringLiteral("ΔAG > 0 (Sắc nét hơn)"),
        QStringLiteral("ΔPIQE < 0 (PIQE tốt hơn)")
    });

    m_comboDetailAlgo = new QComboBox(grpDetail);
    m_lblDetailInfo = new QLabel(QStringLiteral("0 ảnh"), grpDetail);
    m_lblDetailInfo->setStyleSheet(QStringLiteral("color: #9aa0aa;"));

    detBar->addWidget(new QLabel(QStringLiteral("Tìm:"), grpDetail));
    detBar->addWidget(m_editSearch, 1);
    detBar->addWidget(new QLabel(QStringLiteral("Lọc:"), grpDetail));
    detBar->addWidget(m_comboFilter);
    detBar->addWidget(new QLabel(QStringLiteral("Thuật toán:"), grpDetail));
    detBar->addWidget(m_comboDetailAlgo);
    detBar->addWidget(m_lblDetailInfo);
    detLayout->addLayout(detBar);

    m_tblDetail = new QTableWidget(grpDetail);
    m_tblDetail->setObjectName(QStringLiteral("tblDetail"));
    m_tblDetail->setColumnCount(13);
    m_tblDetail->setHorizontalHeaderLabels({
        QStringLiteral("#"),
        QStringLiteral("Tên ảnh"),
        QStringLiteral("Size"),
        QStringLiteral("PSNR ↑"),
        QStringLiteral("IE gốc"),
        QStringLiteral("IE out ↑"),
        QStringLiteral("ΔIE"),
        QStringLiteral("AG gốc"),
        QStringLiteral("AG out ↑"),
        QStringLiteral("ΔAG"),
        QStringLiteral("PIQE ↓"),
        QStringLiteral("ΔPIQE"),
        QStringLiteral("Time (ms) ↓")
    });
    m_tblDetail->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_tblDetail->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_tblDetail->setSelectionMode(QAbstractItemView::SingleSelection);
    m_tblDetail->setAlternatingRowColors(true);
    m_tblDetail->setSortingEnabled(true);
    m_tblDetail->verticalHeader()->setVisible(false);
    m_tblDetail->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    m_tblDetail->horizontalHeader()->setSectionResizeMode(kColDetName, QHeaderView::Stretch);

    detLayout->addWidget(m_tblDetail, 1);

    vSplitter->addWidget(hSplitter);
    vSplitter->addWidget(grpDetail);
    vSplitter->setSizes({340, 360});

    root->addWidget(vSplitter, 1);
}

void DatasetBenchmarkWidget::setupConnections()
{
    connect(m_comboBenchDataset, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DatasetBenchmarkWidget::onBenchDatasetSelected);
    connect(m_btnChooseDataset, &QPushButton::clicked, this, &DatasetBenchmarkWidget::onChooseDataset);
    connect(m_spinMaxImages, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &DatasetBenchmarkWidget::onMaxImagesChanged);
    connect(m_btnRun, &QPushButton::clicked, this, &DatasetBenchmarkWidget::onRunBenchmark);
    connect(m_btnCancel, &QPushButton::clicked, this, &DatasetBenchmarkWidget::onCancelBenchmark);
    connect(m_btnExportCsv, &QPushButton::clicked, this, &DatasetBenchmarkWidget::onExportCsv);
    connect(m_btnCopyLatex, &QPushButton::clicked, this, &DatasetBenchmarkWidget::onCopyLatex);
    connect(m_btnCopyMarkdown, &QPushButton::clicked, this, &DatasetBenchmarkWidget::onCopyMarkdown);
    connect(m_btnSaveChart, &QPushButton::clicked, this, &DatasetBenchmarkWidget::onSaveChartPng);

    connect(m_comboChartType, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DatasetBenchmarkWidget::onChartConfigChanged);
    connect(m_comboChartMetric, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DatasetBenchmarkWidget::onChartConfigChanged);

    connect(m_editSearch, &QLineEdit::textChanged, this, &DatasetBenchmarkWidget::onFilterChanged);
    connect(m_comboFilter, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DatasetBenchmarkWidget::onFilterChanged);
    connect(m_comboDetailAlgo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DatasetBenchmarkWidget::onDetailAlgoChanged);

    connect(m_tblDetail, &QTableWidget::cellDoubleClicked,
            this, &DatasetBenchmarkWidget::onDetailCellDoubleClicked);
    connect(m_watcher, &QFutureWatcher<void>::finished,
            this, &DatasetBenchmarkWidget::onBenchmarkFinished);
}

void DatasetBenchmarkWidget::setAvailableDatasets(
    const std::vector<msgf::DatasetEntry>& datasets,
    const QString& defaultRoot)
{
    m_defaultRoot = defaultRoot;
    const QSignalBlocker blocker(m_comboBenchDataset);
    m_comboBenchDataset->clear();
    for (const auto& ds : datasets) {
        m_comboBenchDataset->addItem(
            QString::fromStdString(ds.displayName),
            QString::fromStdString(ds.folderPath)
        );
    }
    if (!m_datasetDir.isEmpty()) {
        const int idx = m_comboBenchDataset->findData(m_datasetDir);
        if (idx >= 0) {
            m_comboBenchDataset->setCurrentIndex(idx);
        }
    }
}

void DatasetBenchmarkWidget::setDatasetDirectory(const QString& dirPath)
{
    if (dirPath.isEmpty()) {
        return;
    }
    scanDirectory(dirPath);
}

void DatasetBenchmarkWidget::setMsgfParams(const MsgfParams& params)
{
    m_msgfParams = params;
}

bool DatasetBenchmarkWidget::isRunning() const noexcept
{
    return m_isRunning || (m_watcher && m_watcher->isRunning());
}

void DatasetBenchmarkWidget::scanDirectory(const QString& dirPath)
{
    const QDir dir(dirPath);
    if (!dir.exists()) {
        return;
    }

    const int maxLimit = m_spinMaxImages ? m_spinMaxImages->value() : 0;
    const std::vector<std::string> scanned =
        msgf::DatasetManager::scanThermalImages(dirPath.toStdString(), maxLimit);

    m_imageFiles.clear();
    m_imageFiles.reserve(static_cast<int>(scanned.size()));
    for (const std::string& p : scanned) {
        m_imageFiles.append(QString::fromStdString(p));
    }

    m_datasetDir = dir.canonicalPath();
    m_lblDatasetPath->setText(
        QStringLiteral("%1 (%2 ảnh)").arg(m_datasetDir).arg(m_imageFiles.size())
    );
    m_lblDatasetPath->setToolTip(m_datasetDir);
    m_progressBar->setRange(0, std::max(1, static_cast<int>(m_imageFiles.size())));
    m_progressBar->setValue(0);

    if (m_comboBenchDataset) {
        const QSignalBlocker blocker(m_comboBenchDataset);
        const int idx = m_comboBenchDataset->findData(m_datasetDir);
        if (idx >= 0) {
            m_comboBenchDataset->setCurrentIndex(idx);
        } else {
            const QString customLabel = QStringLiteral("Custom: %1 (%2 ảnh)")
                                            .arg(QFileInfo(m_datasetDir).fileName())
                                            .arg(m_imageFiles.size());
            // Update or append custom entry
            int customIdx = -1;
            for (int i = 0; i < m_comboBenchDataset->count(); ++i) {
                if (m_comboBenchDataset->itemText(i).startsWith(QStringLiteral("Custom:"))) {
                    customIdx = i;
                    break;
                }
            }
            if (customIdx >= 0) {
                m_comboBenchDataset->setItemText(customIdx, customLabel);
                m_comboBenchDataset->setItemData(customIdx, m_datasetDir);
                m_comboBenchDataset->setCurrentIndex(customIdx);
            } else {
                m_comboBenchDataset->addItem(customLabel, m_datasetDir);
                m_comboBenchDataset->setCurrentIndex(m_comboBenchDataset->count() - 1);
            }
        }
    }
}

std::vector<msgf::BenchmarkAlgorithmId> DatasetBenchmarkWidget::selectedAlgorithms() const
{
    std::vector<msgf::BenchmarkAlgorithmId> algos;
    algos.push_back(msgf::BenchmarkAlgorithmId::Original);
    if (m_chkHE->isChecked())    algos.push_back(msgf::BenchmarkAlgorithmId::HE);
    if (m_chkGFDDE->isChecked()) algos.push_back(msgf::BenchmarkAlgorithmId::GF_DDE);
    if (m_chkBFDDE->isChecked()) algos.push_back(msgf::BenchmarkAlgorithmId::BF_DDE);
    if (m_chkMSGF->isChecked())  algos.push_back(msgf::BenchmarkAlgorithmId::MSGF_TIR);
    return algos;
}

void DatasetBenchmarkWidget::setRunningState(bool running)
{
    m_isRunning = running;
    m_btnRun->setEnabled(!running);
    m_btnCancel->setEnabled(running);
    m_comboBenchDataset->setEnabled(!running);
    m_btnChooseDataset->setEnabled(!running);
    m_spinMaxImages->setEnabled(!running);
    m_chkHE->setEnabled(!running);
    m_chkGFDDE->setEnabled(!running);
    m_chkBFDDE->setEnabled(!running);
    m_chkMSGF->setEnabled(!running);

    const bool hasResults = !m_summaries.empty();
    m_btnExportCsv->setEnabled(!running && hasResults);
    m_btnCopyLatex->setEnabled(!running && hasResults);
    m_btnCopyMarkdown->setEnabled(!running && hasResults);
    m_btnSaveChart->setEnabled(hasResults);
}

void DatasetBenchmarkWidget::onChooseDataset()
{
    const bool onSample = m_datasetDir.endsWith(QStringLiteral("/sample_images"));
    const QString start = (!m_defaultRoot.isEmpty() && (m_datasetDir.isEmpty() || onSample))
                              ? m_defaultRoot
                              : (m_datasetDir.isEmpty() ? QDir::homePath() : m_datasetDir);
    const QString chosen = QFileDialog::getExistingDirectory(
        this,
        QStringLiteral("Chọn thư mục Dataset ảnh nhiệt (hỗ trợ quét lồng QART_Datasets)"),
        start
    );
    if (!chosen.isEmpty()) {
        scanDirectory(chosen);
    }
}

void DatasetBenchmarkWidget::onBenchDatasetSelected(int index)
{
    if (index < 0 || index >= m_comboBenchDataset->count()) {
        return;
    }
    const QString path = m_comboBenchDataset->itemData(index).toString();
    if (!path.isEmpty()) {
        scanDirectory(path);
    }
}

void DatasetBenchmarkWidget::onMaxImagesChanged(int /*value*/)
{
    if (!isRunning() && !m_datasetDir.isEmpty()) {
        scanDirectory(m_datasetDir);
    }
}

void DatasetBenchmarkWidget::onRunBenchmark()
{
    if (isRunning()) {
        return;
    }
    if (m_imageFiles.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("Dataset Benchmark"),
                                 QStringLiteral("Thư mục hiện tại không có ảnh hợp lệ để benchmark."));
        return;
    }

    const auto algos = selectedAlgorithms();
    if (algos.size() <= 1) {
        QMessageBox::information(this, QStringLiteral("Dataset Benchmark"),
                                 QStringLiteral("Hãy chọn ít nhất 1 thuật toán (ví dụ: MSGF-TIR hoặc HE)."));
        return;
    }

    m_lastOrderedAlgos = algos;
    m_perAlgoRows.clear();
    m_summaries.clear();
    m_tblSummary->setRowCount(0);
    m_tblDetail->setSortingEnabled(false);
    m_tblDetail->setRowCount(0);

    // Populate detail algorithm selector (default to MSGF-TIR if checked)
    {
        const QSignalBlocker blocker(m_comboDetailAlgo);
        m_comboDetailAlgo->clear();
        int msgfIdx = -1;
        for (msgf::BenchmarkAlgorithmId id : algos) {
            if (id == msgf::BenchmarkAlgorithmId::Original) continue;
            const int idx = m_comboDetailAlgo->count();
            m_comboDetailAlgo->addItem(
                QString::fromStdString(msgf::MsgfBaselines::algorithmName(id)),
                static_cast<int>(id)
            );
            if (id == msgf::BenchmarkAlgorithmId::MSGF_TIR) {
                msgfIdx = idx;
            }
        }
        if (msgfIdx >= 0) {
            m_comboDetailAlgo->setCurrentIndex(msgfIdx);
        }
    }

    const int total = static_cast<int>(m_imageFiles.size());
    m_progressBar->setRange(0, total);
    m_progressBar->setValue(0);
    m_lblStatus->setText(QStringLiteral("Đang chạy..."));
    setRunningState(true);

    const std::uint64_t currentRunId = ++m_runId;
    m_cancelFlag = std::make_shared<std::atomic<bool>>(false);
    auto cancelToken = m_cancelFlag;
    const QStringList filesCopy = m_imageFiles;
    const MsgfParams paramsCopy = m_msgfParams;
    QPointer<DatasetBenchmarkWidget> guard(this);

    auto future = QtConcurrent::run([guard, currentRunId, cancelToken, filesCopy, algos, paramsCopy, total]() {
        for (int i = 0; i < total; ++i) {
            if (cancelToken->load(std::memory_order_relaxed)) {
                break;
            }

            const QString& qpath = filesCopy.at(i);
            const cv::Mat rawImg = cv::imread(qpath.toStdString(), cv::IMREAD_UNCHANGED);
            if (rawImg.empty()) {
                QMetaObject::invokeMethod(guard.data(), [guard, currentRunId, i]() {
                    if (!guard || guard->m_runId != currentRunId) return;
                    guard->m_progressBar->setValue(i + 1);
                }, Qt::QueuedConnection);
                continue;
            }

            const cv::Mat orig8U = MsgfAlgorithm::toGray8U(rawImg);
            const MsgfPlaneStats origStats = MsgfMetrics::evaluateSingle(orig8U);
            const QFileInfo fi(qpath);

            auto batch = std::make_shared<PerImageBatchResult>();
            for (msgf::BenchmarkAlgorithmId id : algos) {
                if (cancelToken->load(std::memory_order_relaxed)) {
                    break;
                }

                msgf::ImageBenchmarkRow row;
                row.index = i + 1;
                row.fileName = fi.fileName().toStdString();
                row.fullPath = qpath.toStdString();
                row.width = orig8U.cols;
                row.height = orig8U.rows;
                row.origIE = origStats.ie;
                row.origAG = origStats.ag;
                row.origPIQE = origStats.piqe;

                if (id == msgf::BenchmarkAlgorithmId::Original) {
                    row.psnr = 0.0;
                    row.ie = origStats.ie;
                    row.ag = origStats.ag;
                    row.piqe = origStats.piqe;
                    row.runtimeMs = 0.0;
                } else {
                    const auto res = msgf::MsgfBaselines::runAlgorithm(id, orig8U, paramsCopy);
                    const MsgfMetricResult m = MsgfMetrics::evaluate(orig8U, res.output8U);
                    row.psnr = m.psnr;
                    row.ie = m.ie;
                    row.ag = m.ag;
                    row.piqe = m.piqe;
                    row.runtimeMs = res.elapsedMs;
                }
                batch->rowsByAlgo[id] = std::move(row);
            }

            if (cancelToken->load(std::memory_order_relaxed)) {
                break;
            }

            QMetaObject::invokeMethod(guard.data(), [guard, currentRunId, batch, i, total]() {
                if (!guard || guard->m_runId != currentRunId) {
                    return;
                }
                for (const auto& [id, row] : batch->rowsByAlgo) {
                    guard->m_perAlgoRows[id].push_back(row);
                }

                const auto detailId = static_cast<msgf::BenchmarkAlgorithmId>(
                    guard->m_comboDetailAlgo->currentData().toInt()
                );
                auto it = batch->rowsByAlgo.find(detailId);
                if (it != batch->rowsByAlgo.end()) {
                    guard->appendDetailRow(it->second);
                }

                guard->m_progressBar->setValue(i + 1);
                if (((i + 1) % 5) == 0 || (i + 1) == total) {
                    guard->refreshSummaryTable();
                    guard->refreshChart();
                }
            }, Qt::QueuedConnection);
        }
    });

    m_watcher->setFuture(future);
}

void DatasetBenchmarkWidget::onCancelBenchmark()
{
    if (m_cancelFlag) {
        m_cancelFlag->store(true, std::memory_order_relaxed);
    }
    m_lblStatus->setText(QStringLiteral("Đang hủy..."));
}

void DatasetBenchmarkWidget::onBenchmarkFinished()
{
    refreshSummaryTable();
    refreshChart();
    m_tblDetail->setSortingEnabled(true);
    applyDetailFilters();

    const bool cancelled = m_cancelFlag && m_cancelFlag->load(std::memory_order_relaxed);
    m_lblStatus->setText(cancelled ? QStringLiteral("Đã hủy") : QStringLiteral("Hoàn tất"));
    setRunningState(false);
}

void DatasetBenchmarkWidget::refreshSummaryTable()
{
    m_summaries = msgf::summarizeDatasetBenchmark(m_lastOrderedAlgos, m_perAlgoRows);

    const bool prevSort = m_tblSummary->isSortingEnabled();
    m_tblSummary->setSortingEnabled(false);
    m_tblSummary->setRowCount(static_cast<int>(m_summaries.size()));

    for (int r = 0; r < static_cast<int>(m_summaries.size()); ++r) {
        const auto& s = m_summaries[static_cast<std::size_t>(r)];
        const bool isOrig = (s.algoId == msgf::BenchmarkAlgorithmId::Original);

        auto* nameItem = new QTableWidgetItem(QString::fromStdString(s.algoName));
        if (s.algoId == msgf::BenchmarkAlgorithmId::MSGF_TIR) {
            QFont f = nameItem->font();
            f.setBold(true);
            nameItem->setFont(f);
            nameItem->setForeground(QColor(52, 211, 153));
        }

        auto makeDashItem = []() {
            auto* item = new QTableWidgetItem(QStringLiteral("—"));
            item->setTextAlignment(Qt::AlignCenter);
            return item;
        };

        m_tblSummary->setItem(r, kColSummaryMethod, nameItem);
        m_tblSummary->setItem(r, kColSummaryPsnr,
            isOrig ? makeDashItem() : new NumericTableItem(s.meanPsnr, formatMeanStd(s.meanPsnr, s.stdPsnr)));
        m_tblSummary->setItem(r, kColSummaryIE,
            new NumericTableItem(s.meanIE, formatMeanStd(s.meanIE, s.stdIE)));
        m_tblSummary->setItem(r, kColSummaryAG,
            new NumericTableItem(s.meanAG, formatMeanStd(s.meanAG, s.stdAG)));
        m_tblSummary->setItem(r, kColSummaryPiqe,
            new NumericTableItem(s.meanPIQE, formatMeanStd(s.meanPIQE, s.stdPIQE)));
        m_tblSummary->setItem(r, kColSummaryTime,
            isOrig ? makeDashItem() : new NumericTableItem(s.meanMs, formatMeanStd(s.meanMs, s.stdMs)));

        if (isOrig) {
            m_tblSummary->setItem(r, kColSummaryWinRate, makeDashItem());
        } else {
            auto* winItem = new NumericTableItem(s.winRatePct, 1, QStringLiteral("%"));
            winItem->setToolTip(
                QStringLiteral("Tỉ lệ đứng hạng 1 toàn dataset:\n• PSNR: %1%\n• IE: %2%\n• AG: %3%\n• PIQE: %4%")
                    .arg(s.winRatePSNRPct, 0, 'f', 1)
                    .arg(s.winRateIEPct, 0, 'f', 1)
                    .arg(s.winRateAGPct, 0, 'f', 1)
                    .arg(s.winRatePIQEPct, 0, 'f', 1)
            );
            m_tblSummary->setItem(r, kColSummaryWinRate, winItem);
        }
        m_tblSummary->setRowHeight(r, 24);
    }

    highlightBestAndSecondInColumn(m_tblSummary, kColSummaryPsnr, true);
    highlightBestAndSecondInColumn(m_tblSummary, kColSummaryIE, true);
    highlightBestAndSecondInColumn(m_tblSummary, kColSummaryAG, true);
    highlightBestAndSecondInColumn(m_tblSummary, kColSummaryPiqe, false);
    highlightBestAndSecondInColumn(m_tblSummary, kColSummaryTime, false);
    highlightBestAndSecondInColumn(m_tblSummary, kColSummaryWinRate, true);

    m_tblSummary->setSortingEnabled(prevSort);
}

void DatasetBenchmarkWidget::highlightBestAndSecondInColumn(
    QTableWidget* table,
    int col,
    bool higherIsBetter)
{
    std::vector<std::pair<double, int>> items;
    for (int r = 0; r < table->rowCount(); ++r) {
        auto* item = table->item(r, col);
        if (!item || !item->data(Qt::UserRole).isValid()) {
            continue; // Skip Original "—" rows
        }
        items.emplace_back(item->data(Qt::UserRole).toDouble(), r);
    }
    if (items.empty()) {
        return;
    }

    std::stable_sort(items.begin(), items.end(), [higherIsBetter](const auto& a, const auto& b) {
        return higherIsBetter ? (a.first > b.first) : (a.first < b.first);
    });

    if (auto* bestItem = table->item(items[0].second, col)) {
        QFont f = bestItem->font();
        f.setBold(true);
        bestItem->setFont(f);
        bestItem->setBackground(QColor(46, 160, 67, 75));
    }
    if (items.size() >= 2) {
        if (auto* secondItem = table->item(items[1].second, col)) {
            QFont f = secondItem->font();
            f.setUnderline(true);
            secondItem->setFont(f);
            secondItem->setBackground(QColor(56, 139, 253, 55));
        }
    }
}

void DatasetBenchmarkWidget::appendDetailRow(const msgf::ImageBenchmarkRow& r)
{
    const int rowIdx = m_tblDetail->rowCount();
    m_tblDetail->insertRow(rowIdx);

    auto* idxItem = new NumericTableItem(static_cast<double>(r.index), 0);
    auto* nameItem = new QTableWidgetItem(QString::fromStdString(r.fileName));
    nameItem->setData(Qt::UserRole + 1, QString::fromStdString(r.fullPath));
    nameItem->setToolTip(QString::fromStdString(r.fullPath));

    const QString sizeLabel = QStringLiteral("%1×%2").arg(r.width).arg(r.height);
    auto* sizeItem = new NumericTableItem(static_cast<double>(r.width * r.height), sizeLabel);
    sizeItem->setTextAlignment(Qt::AlignCenter);

    const double dIE = r.ie - r.origIE;
    const double dAG = r.ag - r.origAG;
    const double dPiqe = r.piqe - r.origPIQE;

    auto* deltaIeItem = new NumericTableItem(dIE, formatSignedDelta(dIE));
    deltaIeItem->setForeground(dIE >= 0.0 ? QColor(74, 222, 128) : QColor(248, 113, 113));

    auto* deltaAgItem = new NumericTableItem(dAG, formatSignedDelta(dAG));
    deltaAgItem->setForeground(dAG >= 0.0 ? QColor(74, 222, 128) : QColor(248, 113, 113));

    auto* deltaPiqeItem = new NumericTableItem(dPiqe, formatSignedDelta(dPiqe));
    deltaPiqeItem->setForeground(dPiqe <= 0.0 ? QColor(74, 222, 128) : QColor(248, 113, 113));

    m_tblDetail->setItem(rowIdx, kColDetIdx, idxItem);
    m_tblDetail->setItem(rowIdx, kColDetName, nameItem);
    m_tblDetail->setItem(rowIdx, kColDetSize, sizeItem);
    m_tblDetail->setItem(rowIdx, kColDetPsnr, new NumericTableItem(r.psnr, 2));
    m_tblDetail->setItem(rowIdx, kColDetOrigIE, new NumericTableItem(r.origIE, 2));
    m_tblDetail->setItem(rowIdx, kColDetOutIE, new NumericTableItem(r.ie, 2));
    m_tblDetail->setItem(rowIdx, kColDetDeltaIE, deltaIeItem);
    m_tblDetail->setItem(rowIdx, kColDetOrigAG, new NumericTableItem(r.origAG, 2));
    m_tblDetail->setItem(rowIdx, kColDetOutAG, new NumericTableItem(r.ag, 2));
    m_tblDetail->setItem(rowIdx, kColDetDeltaAG, deltaAgItem);
    m_tblDetail->setItem(rowIdx, kColDetOutPiqe, new NumericTableItem(r.piqe, 2));
    m_tblDetail->setItem(rowIdx, kColDetDeltaPiqe, deltaPiqeItem);
    m_tblDetail->setItem(rowIdx, kColDetTime, new NumericTableItem(r.runtimeMs, 2));
    m_tblDetail->setRowHeight(rowIdx, 22);

    const QString query = m_editSearch->text().trimmed();
    const int filterMode = m_comboFilter->currentIndex();
    bool match = query.isEmpty() || nameItem->text().contains(query, Qt::CaseInsensitive);
    if (match && filterMode > 0) {
        switch (filterMode) {
        case 1: match = (dIE > 0.0); break;
        case 2: match = (dIE < 0.0); break;
        case 3: match = (dAG > 0.0); break;
        case 4: match = (dPiqe < 0.0); break;
        default: break;
        }
    }
    m_tblDetail->setRowHidden(rowIdx, !match);
    m_lblDetailInfo->setText(QStringLiteral("%1 ảnh").arg(m_tblDetail->rowCount()));
}

void DatasetBenchmarkWidget::rebuildDetailTable()
{
    const bool prevSort = m_tblDetail->isSortingEnabled();
    m_tblDetail->setSortingEnabled(false);
    m_tblDetail->setRowCount(0);

    if (m_comboDetailAlgo->count() > 0) {
        const auto id = static_cast<msgf::BenchmarkAlgorithmId>(
            m_comboDetailAlgo->currentData().toInt()
        );
        auto it = m_perAlgoRows.find(id);
        if (it != m_perAlgoRows.end()) {
            for (const auto& r : it->second) {
                appendDetailRow(r);
            }
        }
    }

    m_tblDetail->setSortingEnabled(prevSort);
    applyDetailFilters();
}

void DatasetBenchmarkWidget::applyDetailFilters()
{
    const QString query = m_editSearch->text().trimmed();
    const int filterMode = m_comboFilter->currentIndex();
    int visibleCount = 0;
    const int total = m_tblDetail->rowCount();

    for (int r = 0; r < total; ++r) {
        auto* nameItem = m_tblDetail->item(r, kColDetName);
        auto* dIeItem = m_tblDetail->item(r, kColDetDeltaIE);
        auto* dAgItem = m_tblDetail->item(r, kColDetDeltaAG);
        auto* dPiqeItem = m_tblDetail->item(r, kColDetDeltaPiqe);

        bool match = true;
        if (!query.isEmpty() && nameItem) {
            match = nameItem->text().contains(query, Qt::CaseInsensitive);
        }
        if (match && filterMode > 0) {
            const double dIE = dIeItem ? dIeItem->data(Qt::UserRole).toDouble() : 0.0;
            const double dAG = dAgItem ? dAgItem->data(Qt::UserRole).toDouble() : 0.0;
            const double dPiqe = dPiqeItem ? dPiqeItem->data(Qt::UserRole).toDouble() : 0.0;
            switch (filterMode) {
            case 1: match = (dIE > 0.0); break;
            case 2: match = (dIE < 0.0); break;
            case 3: match = (dAG > 0.0); break;
            case 4: match = (dPiqe < 0.0); break;
            default: break;
            }
        }

        m_tblDetail->setRowHidden(r, !match);
        if (match) {
            visibleCount++;
        }
    }

    m_lblDetailInfo->setText(QStringLiteral("%1 / %2 ảnh").arg(visibleCount).arg(total));
}

void DatasetBenchmarkWidget::refreshChart()
{
    auto* chart = new QChart();
    chart->setTheme(QChart::ChartThemeDark);
    chart->setBackgroundBrush(QColor(28, 28, 30));
    chart->setAnimationOptions(QChart::NoAnimation);

    const int chartType = m_comboChartType->currentIndex();
    const QString metricName = m_comboChartMetric->currentText();

    if (m_summaries.empty()) {
        chart->setTitle(QStringLiteral("Chạy Benchmark để hiển thị biểu đồ phân tích Dataset"));
        auto* old = m_chartView->chart();
        m_chartView->setChart(chart);
        if (old) old->deleteLater();
        return;
    }

    if (chartType == 0) {
        // 1. Grouped Bar Chart (QBarSeries)
        auto* series = new QBarSeries();
        auto* barSet = new QBarSet(metricName);
        barSet->setColor(QColor(59, 130, 246));
        QStringList categories;
        double maxVal = 0.0;

        for (const auto& s : m_summaries) {
            const bool isOrig = (s.algoId == msgf::BenchmarkAlgorithmId::Original);
            if (isOrig && (metricName == QStringLiteral("PSNR (dB)") || metricName == QStringLiteral("Time (ms)"))) {
                continue;
            }
            categories << QString::fromStdString(s.algoName);
            const double val = extractMetricValue(s, metricName);
            *barSet << val;
            maxVal = std::max(maxVal, val);
        }

        series->append(barSet);
        series->setLabelsVisible(true);
        series->setLabelsFormat(QStringLiteral("@value"));

        chart->addSeries(series);
        chart->setTitle(QStringLiteral("So sánh trung bình toàn Dataset: ") + metricName);
        chart->legend()->setVisible(false);

        auto* axisX = new QBarCategoryAxis();
        axisX->append(categories);
        chart->addAxis(axisX, Qt::AlignBottom);
        series->attachAxis(axisX);

        auto* axisY = new QValueAxis();
        axisY->setRange(0.0, maxVal * 1.18 + 1e-3);
        axisY->setTitleText(metricName);
        chart->addAxis(axisY, Qt::AlignLeft);
        series->attachAxis(axisY);
    } else if (chartType == 1) {
        // 2. Per-Image Trend (QLineSeries)
        double maxX = 1.0;
        double minY = 1e18;
        double maxY = -1e18;

        auto* axisX = new QValueAxis();
        axisX->setTitleText(QStringLiteral("Thứ tự ảnh (#)"));
        axisX->setLabelFormat(QStringLiteral("%d"));

        auto* axisY = new QValueAxis();
        axisY->setTitleText(metricName);

        chart->addAxis(axisX, Qt::AlignBottom);
        chart->addAxis(axisY, Qt::AlignLeft);
        chart->setTitle(QStringLiteral("Biến thiên theo từng ảnh: ") + metricName);

        for (msgf::BenchmarkAlgorithmId id : m_lastOrderedAlgos) {
            const bool isOrig = (id == msgf::BenchmarkAlgorithmId::Original);
            if (isOrig && (metricName == QStringLiteral("PSNR (dB)") || metricName == QStringLiteral("Time (ms)"))) {
                continue;
            }
            auto it = m_perAlgoRows.find(id);
            if (it == m_perAlgoRows.end() || it->second.empty()) {
                continue;
            }

            auto* line = new QLineSeries();
            line->setName(QString::fromStdString(msgf::MsgfBaselines::algorithmName(id)));
            QPen pen(algoColor(id));
            pen.setWidth(id == msgf::BenchmarkAlgorithmId::MSGF_TIR ? 2 : 1);
            line->setPen(pen);

            for (const auto& r : it->second) {
                const double x = static_cast<double>(r.index);
                const double y = extractRowMetricValue(r, metricName);
                line->append(x, y);
                maxX = std::max(maxX, x);
                minY = std::min(minY, y);
                maxY = std::max(maxY, y);
            }

            chart->addSeries(line);
            line->attachAxis(axisX);
            line->attachAxis(axisY);
        }

        if (minY > maxY) {
            minY = 0.0;
            maxY = 1.0;
        }
        const double pad = std::max(0.1, (maxY - minY) * 0.12);
        axisX->setRange(1.0, std::max(2.0, maxX));
        axisY->setRange(std::max(0.0, minY - pad), maxY + pad);
        chart->legend()->setAlignment(Qt::AlignBottom);
    } else {
        // 3. Speed vs Quality Trade-off Scatter (QScatterSeries)
        const QString yMetric = (metricName == QStringLiteral("Time (ms)"))
                                    ? QStringLiteral("AG")
                                    : metricName;
        double maxX = 1.0;
        double minY = 1e18;
        double maxY = -1e18;

        auto* axisX = new QValueAxis();
        axisX->setTitleText(QStringLiteral("Runtime (ms) ← nhanh hơn"));

        auto* axisY = new QValueAxis();
        axisY->setTitleText(yMetric);

        chart->addAxis(axisX, Qt::AlignBottom);
        chart->addAxis(axisY, Qt::AlignLeft);
        chart->setTitle(QStringLiteral("Đánh đổi Tốc độ vs Chất lượng (Runtime vs %1)").arg(yMetric));

        for (msgf::BenchmarkAlgorithmId id : m_lastOrderedAlgos) {
            if (id == msgf::BenchmarkAlgorithmId::Original) {
                continue;
            }
            auto it = m_perAlgoRows.find(id);
            if (it == m_perAlgoRows.end() || it->second.empty()) {
                continue;
            }

            auto* scatter = new QScatterSeries();
            scatter->setName(QString::fromStdString(msgf::MsgfBaselines::algorithmName(id)));
            scatter->setMarkerSize(id == msgf::BenchmarkAlgorithmId::MSGF_TIR ? 9.0 : 7.0);
            scatter->setColor(algoColor(id));
            scatter->setBorderColor(QColor(24, 24, 28));

            for (const auto& r : it->second) {
                const double x = r.runtimeMs;
                const double y = extractRowMetricValue(r, yMetric);
                scatter->append(x, y);
                maxX = std::max(maxX, x);
                minY = std::min(minY, y);
                maxY = std::max(maxY, y);
            }

            chart->addSeries(scatter);
            scatter->attachAxis(axisX);
            scatter->attachAxis(axisY);
        }

        if (minY > maxY) {
            minY = 0.0;
            maxY = 1.0;
        }
        const double padY = std::max(0.1, (maxY - minY) * 0.12);
        axisX->setRange(0.0, maxX * 1.15 + 0.5);
        axisY->setRange(std::max(0.0, minY - padY), maxY + padY);
        chart->legend()->setAlignment(Qt::AlignBottom);
    }

    auto* old = m_chartView->chart();
    m_chartView->setChart(chart);
    if (old) {
        old->deleteLater();
    }
}

void DatasetBenchmarkWidget::onChartConfigChanged()
{
    refreshChart();
}

void DatasetBenchmarkWidget::onFilterChanged()
{
    applyDetailFilters();
}

void DatasetBenchmarkWidget::onDetailAlgoChanged(int /*index*/)
{
    rebuildDetailTable();
}

void DatasetBenchmarkWidget::onDetailCellDoubleClicked(int row, int /*column*/)
{
    auto* nameItem = m_tblDetail->item(row, kColDetName);
    if (!nameItem) {
        return;
    }
    const QString fullPath = nameItem->data(Qt::UserRole + 1).toString();
    if (!fullPath.isEmpty()) {
        emit imageActivated(fullPath);
    }
}

void DatasetBenchmarkWidget::onSaveChartPng()
{
    const QString path = QFileDialog::getSaveFileName(
        this,
        QStringLiteral("Lưu biểu đồ Benchmark ra ảnh PNG"),
        QDir::homePath() + QStringLiteral("/msgf_benchmark_chart.png"),
        QStringLiteral("PNG Image (*.png)")
    );
    if (path.isEmpty()) {
        return;
    }

    const QPixmap pix = m_chartView->grab();
    if (!pix.save(path, "PNG", 100)) {
        QMessageBox::warning(this, QStringLiteral("Lưu biểu đồ"),
                             QStringLiteral("Không thể ghi file:\n%1").arg(path));
    }
}

void DatasetBenchmarkWidget::onExportCsv()
{
    if (m_summaries.empty()) {
        return;
    }

    const QString dirPath = QFileDialog::getExistingDirectory(
        this,
        QStringLiteral("Chọn thư mục lưu dataset_summary.csv & dataset_detail.csv"),
        QDir::homePath()
    );
    if (dirPath.isEmpty()) {
        return;
    }

    const QDir dir(dirPath);
    const std::string sumCsv = msgf::exportSummaryToCsvUtf8Bom(m_summaries);
    const std::string detCsv = msgf::exportDetailToCsvUtf8Bom(m_lastOrderedAlgos, m_perAlgoRows);

    auto writeAll = [](const QString& filePath, const std::string& bytes) -> bool {
        QFile f(filePath);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            return false;
        }
        const qint64 written = f.write(bytes.data(), static_cast<qint64>(bytes.size()));
        return written == static_cast<qint64>(bytes.size());
    };

    const QString sumPath = dir.filePath(QStringLiteral("dataset_summary.csv"));
    const QString detPath = dir.filePath(QStringLiteral("dataset_detail.csv"));
    if (writeAll(sumPath, sumCsv) && writeAll(detPath, detCsv)) {
        QMessageBox::information(
            this,
            QStringLiteral("Xuất CSV thành công"),
            QStringLiteral("Đã xuất 2 file CSV chuẩn UTF-8 BOM:\n• %1\n• %2").arg(sumPath, detPath)
        );
    } else {
        QMessageBox::warning(this, QStringLiteral("Xuất CSV"),
                             QStringLiteral("Lỗi ghi file CSV trong thư mục:\n%1").arg(dirPath));
    }
}

void DatasetBenchmarkWidget::onCopyLatex()
{
    if (m_summaries.empty()) {
        return;
    }
    const std::string tex = msgf::exportSummaryToLatex(m_summaries);
    QApplication::clipboard()->setText(QString::fromStdString(tex));
    m_lblStatus->setText(QStringLiteral("Đã copy LaTeX"));
}

void DatasetBenchmarkWidget::onCopyMarkdown()
{
    if (m_summaries.empty()) {
        return;
    }
    const std::string md = msgf::exportSummaryToMarkdown(m_summaries);
    QApplication::clipboard()->setText(QString::fromStdString(md));
    m_lblStatus->setText(QStringLiteral("Đã copy Markdown"));
}
