#include "ui/N5Form.h"
#include "ui_N5Form.h"

#include "core/MsgfAlgorithm.h"
#include "ui/DatasetBenchmarkWidget.h"
#include "ui/ImageCanvas.h"

#include <QApplication>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHeaderView>
#include <QMessageBox>
#include <QSettings>
#include <QSignalBlocker>
#include <QTableWidgetItem>
#include <QtConcurrent/QtConcurrentRun>

#include <opencv2/imgcodecs.hpp>

#include <exception>

namespace {

QString formatDeltaN5(double delta, int precision = 2, bool lowerIsBetter = false)
{
    const QString sign = (delta >= 0.0) ? QStringLiteral("+") : QStringLiteral("");
    const QString arrow = (lowerIsBetter ? (delta <= 0.0 ? QStringLiteral("✓") : QStringLiteral(""))
                                         : (delta >= 0.0 ? QStringLiteral("✓") : QStringLiteral("")));
    return QStringLiteral("%1%2 %3").arg(sign).arg(delta, 0, 'f', precision).arg(arrow).trimmed();
}

} // namespace

N5Form::N5Form(QWidget* parent)
    : QWidget(parent)
    , ui(std::make_unique<Ui::N5Form>())
    , m_debounceTimer(new QTimer(this))
    , m_watcher(new QFutureWatcher<WorkerResult>(this))
{
    ui->setupUi(this);

    m_debounceTimer->setSingleShot(true);
    m_debounceTimer->setInterval(60);

    applyTheme();
    setupAblationTable();
    setupConnections();
    updateFreezeControls();

    ui->mainSplitter->setSizes({340, 790, 350});
    ui->benchmarkWidget->setDefaultFocusAlgorithm(msgf::BenchmarkAlgorithmId::N5);

    ui->canvasInput->setTitle(QStringLiteral("Prepared Input (P)"));
    ui->canvasOutput->setTitle(QStringLiteral("N5 Post-Half Output (Y)"));
    ui->canvasB1->setTitle(QStringLiteral("(1) Guided Base B1 (w1)"));
    ui->canvasR1->setTitle(QStringLiteral("(2) Residual R1 = x - B1 (norm)"));
    ui->canvasC->setTitle(QStringLiteral("(3) CLAHE Base C"));
    ui->canvasU->setTitle(QStringLiteral("(4) Stage-1 Quantized U"));
    ui->canvasB2->setTitle(QStringLiteral("(5) Fine Base B2 (w2)"));
    ui->canvasR2->setTitle(QStringLiteral("(6) Fine Residual R2 (norm)"));

    initializeDatasets();
}

N5Form::~N5Form()
{
    if (m_debounceTimer->isActive()) {
        m_debounceTimer->stop();
    }
    if (m_watcher->isRunning()) {
        m_watcher->waitForFinished();
    }
}

void N5Form::applyTheme()
{
    setStyleSheet(QStringLiteral(R"(
        QWidget#N5Form {
            background-color: #1e1f22;
            color: #e4e6eb;
            font-size: 12px;
        }
        QGroupBox {
            border: 1px solid #383a40;
            border-radius: 6px;
            margin-top: 10px;
            padding: 8px 8px 6px 8px;
            font-weight: 600;
            color: #cfd4dc;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            color: #f472b6;
        }
        QPushButton {
            background-color: #2c2f36;
            border: 1px solid #43464f;
            border-radius: 4px;
            padding: 4px 10px;
            color: #e6e8ed;
        }
        QPushButton:hover {
            background-color: #363a43;
            border-color: #f472b6;
        }
        QPushButton:pressed {
            background-color: #252830;
        }
        QPushButton#btnRun {
            background-color: #9d174d;
            border-color: #db2777;
            font-weight: 600;
        }
        QPushButton#btnRun:hover {
            background-color: #be185d;
        }
        QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #25272c;
            border: 1px solid #3f424b;
            border-radius: 4px;
            padding: 2px 6px;
            color: #eceff4;
        }
        QTabWidget::pane {
            border: 1px solid #383a40;
            border-radius: 4px;
            background-color: #1c1c1e;
        }
        QTabBar::tab {
            background-color: #27292e;
            color: #b8bec9;
            padding: 6px 14px;
            border: 1px solid #383a40;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #1c1c1e;
            color: #ffffff;
            border-bottom: 2px solid #f472b6;
        }
        QTableWidget {
            background-color: #23252a;
            alternate-background-color: #282b31;
            gridline-color: #383a40;
            color: #e4e6eb;
            border: 1px solid #383a40;
            border-radius: 4px;
            font-size: 11px;
        }
        QHeaderView::section {
            background-color: #2c2f36;
            color: #cfd4dc;
            padding: 4px;
            border: none;
            border-bottom: 1px solid #3f424b;
            font-weight: 600;
        }
    )"));
}

void N5Form::setupAblationTable()
{
    auto* tbl = ui->tblAblation;
    tbl->setColumnCount(5);
    tbl->setHorizontalHeaderLabels({
        QStringLiteral("Stage"),
        QStringLiteral("PSNR ↑"),
        QStringLiteral("IE ↑"),
        QStringLiteral("AG ↑"),
        QStringLiteral("PIQE ↓")
    });
    tbl->setRowCount(4);
    tbl->verticalHeader()->setVisible(false);

    const QStringList labels = {
        QStringLiteral("1. Input (P)"),
        QStringLiteral("2. CLAHE Base (C)"),
        QStringLiteral("3. Stage-1 (U)"),
        QStringLiteral("4. Final N5 (Y)")
    };
    for (int r = 0; r < 4; ++r) {
        auto* item = new QTableWidgetItem(labels[r]);
        if (r == 3) {
            QFont f = item->font();
            f.setBold(true);
            item->setFont(f);
            item->setForeground(QColor(244, 114, 182));
        }
        tbl->setItem(r, 0, item);
        for (int c = 1; c < 5; ++c) {
            auto* v = new QTableWidgetItem(QStringLiteral("—"));
            v->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
            tbl->setItem(r, c, v);
        }
        tbl->setRowHeight(r, 24);
    }

    auto* hh = tbl->horizontalHeader();
    hh->setSectionResizeMode(0, QHeaderView::Stretch);
    for (int col = 1; col < 5; ++col) {
        hh->setSectionResizeMode(col, QHeaderView::ResizeToContents);
    }
}

void N5Form::setupConnections()
{
    connect(ui->btnLoadImage, &QPushButton::clicked, this, &N5Form::onLoadImage);
    connect(ui->btnOpenFolder, &QPushButton::clicked, this, &N5Form::onOpenFolder);
    connect(ui->btnSetRoot, &QPushButton::clicked, this, &N5Form::onSetDatasetRoot);
    connect(ui->comboDataset, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &N5Form::onDatasetSelected);
    connect(ui->comboScene, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &N5Form::onSceneSelected);
    connect(ui->btnPrev, &QPushButton::clicked, this, &N5Form::onPrevImage);
    connect(ui->btnNext, &QPushButton::clicked, this, &N5Form::onNextImage);

    connect(ui->chkFreezeAccepted, &QCheckBox::toggled, this, &N5Form::onFreezeToggled);
    connect(ui->btnRun, &QPushButton::clicked, this, &N5Form::onRunClicked);
    connect(ui->btnReset, &QPushButton::clicked, this, &N5Form::onResetAccepted);
    connect(ui->btnSaveOutput, &QPushButton::clicked, this, &N5Form::onSaveOutput);
    connect(ui->btnExportStages, &QPushButton::clicked, this, &N5Form::onExportStages);

    for (QSpinBox* sb : {ui->spinW1, ui->spinTileGrid, ui->spinW2}) {
        connect(sb, QOverload<int>::of(&QSpinBox::valueChanged),
                this, &N5Form::onParamEdited);
    }
    for (QDoubleSpinBox* dsb : {ui->spinClipLimit, ui->spinPostHalf}) {
        connect(dsb, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this, &N5Form::onParamEdited);
    }

    connect(m_debounceTimer, &QTimer::timeout, this, &N5Form::startBackgroundRun);
    connect(m_watcher, &QFutureWatcher<WorkerResult>::finished,
            this, &N5Form::onWorkerFinished);

    for (ImageCanvas* canvas : allCanvases()) {
        connect(canvas, &ImageCanvas::viewChanged, this, &N5Form::onSyncViewChanged);
        connect(canvas, &ImageCanvas::viewReset, this, &N5Form::onSyncViewReset);
    }

    connect(ui->benchmarkWidget, &DatasetBenchmarkWidget::imageActivated,
            this, [this](const QString& filePath) {
                if (loadImageFile(filePath)) {
                    ui->viewerTabs->setCurrentIndex(0);
                }
            });
}

std::vector<ImageCanvas*> N5Form::allCanvases() const
{
    return {
        ui->canvasInput,
        ui->canvasOutput,
        ui->canvasB1,
        ui->canvasR1,
        ui->canvasC,
        ui->canvasU,
        ui->canvasB2,
        ui->canvasR2
    };
}

void N5Form::initializeDatasets()
{
    ui->benchmarkWidget->setN5Params(currentParams());
    m_sampleDir = locateSampleDirectory();

    const QSettings settings(QStringLiteral("MSGF_Test"), QStringLiteral("MSGF-TIR"));
    const QString savedRoot = settings.value(QStringLiteral("datasetRoot")).toString();
    refreshDatasets(savedRoot);
}

void N5Form::refreshDatasets(const QString& rootOverride)
{
    const std::string detected = msgf::DatasetManager::detectDatasetRoot(rootOverride.toStdString());
    m_datasetRoot = QString::fromStdString(detected);
    m_datasets = msgf::DatasetManager::discoverDatasets(
        detected,
        m_sampleDir.toStdString()
    );

    {
        const QSignalBlocker blocker(ui->comboDataset);
        ui->comboDataset->clear();
        for (const auto& ds : m_datasets) {
            ui->comboDataset->addItem(
                QString::fromStdString(ds.displayName),
                QString::fromStdString(ds.folderPath)
            );
        }
    }

    ui->benchmarkWidget->setAvailableDatasets(m_datasets, m_datasetRoot);

    if (!m_datasets.empty()) {
        const QSignalBlocker blocker(ui->comboDataset);
        ui->comboDataset->setCurrentIndex(0);
        const QString firstPath = QString::fromStdString(m_datasets[0].folderPath);
        loadFolderImages(firstPath);
        ui->benchmarkWidget->setDatasetDirectory(firstPath);
    } else if (!m_sampleDir.isEmpty()) {
        loadFolderImages(m_sampleDir);
        ui->benchmarkWidget->setDatasetDirectory(m_sampleDir);
    } else {
        updateNavigationButtons();
    }
}

QString N5Form::locateSampleDirectory()
{
    const QString appDir = QCoreApplication::applicationDirPath();
    const QStringList candidates = {
        QDir::currentPath() + QStringLiteral("/sample_images"),
        appDir + QStringLiteral("/sample_images"),
        appDir + QStringLiteral("/../sample_images"),
        appDir + QStringLiteral("/../../sample_images")
    };
    for (const QString& cand : candidates) {
        const QDir d(cand);
        if (d.exists() && !d.entryList({QStringLiteral("*.png"), QStringLiteral("*.jpg")}, QDir::Files).isEmpty()) {
            return d.canonicalPath();
        }
    }
    return {};
}

void N5Form::loadFolderImages(const QString& dirPath)
{
    const QDir dir(dirPath);
    if (!dir.exists()) {
        return;
    }

    const std::vector<std::string> scanned = msgf::DatasetManager::scanThermalImages(dirPath.toStdString());
    if (scanned.empty()) {
        QMessageBox::information(
            this,
            QStringLiteral("Open Folder"),
            QStringLiteral("Không tìm thấy ảnh nhiệt hợp lệ trong thư mục:\n%1").arg(dirPath)
        );
        return;
    }

    const QDir rootDir(dir.canonicalPath());
    m_imageFiles.clear();
    m_imageFiles.reserve(scanned.size());

    ui->comboScene->setUpdatesEnabled(false);
    {
        const QSignalBlocker blocker(ui->comboScene);
        ui->comboScene->clear();

        for (const std::string& stdPath : scanned) {
            const QString qPath = QString::fromStdString(stdPath);
            const QFileInfo fi(qPath);
            const QString fname = fi.fileName();
            m_imageFiles.push_back(qPath);

            QString displayName = rootDir.relativeFilePath(qPath);
            if (displayName.startsWith(QStringLiteral("../"))) {
                displayName = fname;
            }
            if (fname == QStringLiteral("scene1_original.png")) {
                displayName = QStringLiteral("Scene 1 (iRay T3-Pro)");
            } else if (fname == QStringLiteral("scene2_original.png")) {
                displayName = QStringLiteral("Scene 2 (FLIR Vehicle)");
            } else if (fname == QStringLiteral("scene3_original.png")) {
                displayName = QStringLiteral("Scene 3 (FLIR Pedestrian)");
            }
            ui->comboScene->addItem(displayName, qPath);
        }
    }
    ui->comboScene->setUpdatesEnabled(true);

    m_currentFileIndex = m_imageFiles.empty() ? -1 : 0;
    updateNavigationButtons();
    if (m_currentFileIndex >= 0) {
        const QSignalBlocker blocker(ui->comboScene);
        ui->comboScene->setCurrentIndex(0);
        loadImageFile(m_imageFiles[0]);
    }
}

void N5Form::updateNavigationButtons()
{
    const bool hasMultiple = m_imageFiles.size() > 1;
    ui->btnPrev->setEnabled(hasMultiple && m_currentFileIndex > 0);
    ui->btnNext->setEnabled(hasMultiple && m_currentFileIndex >= 0
                            && m_currentFileIndex + 1 < static_cast<int>(m_imageFiles.size()));
}

void N5Form::updateFreezeControls()
{
    const bool editable = !ui->chkFreezeAccepted->isChecked();
    ui->spinW1->setEnabled(editable);
    ui->spinClipLimit->setEnabled(editable);
    ui->spinTileGrid->setEnabled(editable);
    ui->spinW2->setEnabled(editable);
    ui->spinPostHalf->setEnabled(editable);
}

msgf::N5Params N5Form::currentParams() const
{
    msgf::N5Params p;
    p.stage1Window = ui->spinW1->value();
    p.claheClipLimit = ui->spinClipLimit->value();
    p.claheTileGrid = ui->spinTileGrid->value();
    p.stage2Window = ui->spinW2->value();
    p.postHalfAttenuation = ui->spinPostHalf->value();
    return p;
}

void N5Form::setParams(const msgf::N5Params& p)
{
    m_isUpdatingControls = true;
    ui->spinW1->setValue(p.stage1Window);
    ui->spinClipLimit->setValue(p.claheClipLimit);
    ui->spinTileGrid->setValue(p.claheTileGrid);
    ui->spinW2->setValue(p.stage2Window);
    ui->spinPostHalf->setValue(p.postHalfAttenuation);
    m_isUpdatingControls = false;

    ui->benchmarkWidget->setN5Params(p);
    scheduleProcessing(true);
}

void N5Form::setInputImage(const cv::Mat& image, const QString& label)
{
    if (image.empty()) {
        return;
    }
    m_inputMat = image.clone();
    m_currentLabel = label;

    ui->lblImageInfo->setText(
        QStringLiteral("%1 (%2×%3, %4 ch)")
            .arg(label)
            .arg(m_inputMat.cols)
            .arg(m_inputMat.rows)
            .arg(m_inputMat.channels())
    );
    scheduleProcessing(true);
}

bool N5Form::loadImageFile(const QString& filePath)
{
    const cv::Mat loaded = cv::imread(filePath.toStdString(), cv::IMREAD_UNCHANGED);
    if (loaded.empty()) {
        QMessageBox::warning(this, QStringLiteral("Load Image"),
                             QStringLiteral("Failed to read image:\n%1").arg(filePath));
        return false;
    }

    const QFileInfo fi(filePath);
    m_inputMat = loaded;
    m_currentLabel = fi.fileName();

    const QString depthLabel = (loaded.depth() == CV_16U) ? QStringLiteral("16-bit") : QStringLiteral("8-bit");
    ui->lblImageInfo->setText(
        QStringLiteral("%1 (%2×%3, %4)")
            .arg(m_currentLabel)
            .arg(loaded.cols)
            .arg(loaded.rows)
            .arg(depthLabel)
    );

    scheduleProcessing(true);
    return true;
}

void N5Form::onLoadImage()
{
    const QString startDir = !m_datasetRoot.isEmpty()
                                 ? m_datasetRoot
                                 : (m_sampleDir.isEmpty() ? QDir::homePath() : m_sampleDir);
    const QString path = QFileDialog::getOpenFileName(
        this,
        QStringLiteral("Open Thermal Infrared Image"),
        startDir,
        QStringLiteral("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
    );
    if (!path.isEmpty()) {
        loadImageFile(path);
    }
}

void N5Form::onOpenFolder()
{
    const QString startDir = !m_datasetRoot.isEmpty()
                                 ? m_datasetRoot
                                 : (m_sampleDir.isEmpty() ? QDir::homePath() : m_sampleDir);
    const QString dirPath = QFileDialog::getExistingDirectory(
        this,
        QStringLiteral("Select Folder of Thermal Images (or QART_Datasets subfolder)"),
        startDir
    );
    if (!dirPath.isEmpty()) {
        loadFolderImages(dirPath);
        if (!ui->benchmarkWidget->isRunning()) {
            ui->benchmarkWidget->setDatasetDirectory(dirPath);
        }
    }
}

void N5Form::onSetDatasetRoot()
{
    const QString startDir = !m_datasetRoot.isEmpty() ? m_datasetRoot : QDir::homePath();
    const QString dirPath = QFileDialog::getExistingDirectory(
        this,
        QStringLiteral("Select Dataset Root Folder (e.g. QART_Datasets)"),
        startDir
    );
    if (dirPath.isEmpty()) {
        return;
    }
    QSettings settings(QStringLiteral("MSGF_Test"), QStringLiteral("MSGF-TIR"));
    settings.setValue(QStringLiteral("datasetRoot"), dirPath);
    refreshDatasets(dirPath);
}

void N5Form::onDatasetSelected(int index)
{
    if (index < 0 || index >= ui->comboDataset->count()) {
        return;
    }
    const QString folderPath = ui->comboDataset->itemData(index).toString();
    if (!folderPath.isEmpty()) {
        loadFolderImages(folderPath);
        if (!ui->benchmarkWidget->isRunning()) {
            ui->benchmarkWidget->setDatasetDirectory(folderPath);
        }
    }
}

void N5Form::onSceneSelected(int index)
{
    if (index < 0 || index >= static_cast<int>(m_imageFiles.size())) {
        return;
    }
    m_currentFileIndex = index;
    updateNavigationButtons();
    loadImageFile(m_imageFiles[static_cast<std::size_t>(index)]);
}

void N5Form::onPrevImage()
{
    if (m_currentFileIndex > 0) {
        ui->comboScene->setCurrentIndex(m_currentFileIndex - 1);
    }
}

void N5Form::onNextImage()
{
    if (m_currentFileIndex >= 0
        && m_currentFileIndex + 1 < static_cast<int>(m_imageFiles.size())) {
        ui->comboScene->setCurrentIndex(m_currentFileIndex + 1);
    }
}

void N5Form::onFreezeToggled(bool checked)
{
    updateFreezeControls();
    if (checked) {
        setParams(msgf::N5Params{});
    }
}

void N5Form::onParamEdited()
{
    if (m_isUpdatingControls) {
        return;
    }
    ui->benchmarkWidget->setN5Params(currentParams());
    if (ui->chkAutoUpdate->isChecked()) {
        scheduleProcessing(false);
    }
}

void N5Form::onRunClicked()
{
    scheduleProcessing(true);
}

void N5Form::onResetAccepted()
{
    {
        const QSignalBlocker blocker(ui->chkFreezeAccepted);
        ui->chkFreezeAccepted->setChecked(true);
    }
    updateFreezeControls();
    setParams(msgf::N5Params{});
}

void N5Form::scheduleProcessing(bool immediate)
{
    if (m_inputMat.empty()) {
        return;
    }
    if (immediate) {
        m_debounceTimer->stop();
        startBackgroundRun();
    } else {
        m_debounceTimer->start();
    }
}

void N5Form::startBackgroundRun()
{
    if (m_inputMat.empty()) {
        return;
    }

    const std::uint64_t gen = ++m_generation;
    const cv::Mat imgCopy = m_inputMat.clone();
    const msgf::N5Params params = currentParams();

    ui->lblStatusVal->setText(QStringLiteral("Processing..."));

    auto future = QtConcurrent::run([imgCopy, params, gen]() -> WorkerResult {
        WorkerResult res;
        res.generation = gen;
        try {
            res.outputs = msgf::N5Algorithm::run(imgCopy, params);
            res.finalMetrics = MsgfMetrics::evaluate(res.outputs.input8U, res.outputs.output8U);
            res.claheMetrics = MsgfMetrics::evaluate(res.outputs.input8U, res.outputs.C_clahe8U);
            res.stage1Metrics = MsgfMetrics::evaluate(res.outputs.input8U, res.outputs.U_stage18U);
            res.ok = true;
        } catch (const std::exception& e) {
            res.ok = false;
            res.errorText = QString::fromUtf8(e.what());
        }
        return res;
    });
    m_watcher->setFuture(future);
}

void N5Form::onWorkerFinished()
{
    const WorkerResult res = m_watcher->result();
    if (res.generation != m_generation) {
        return;
    }
    if (!res.ok) {
        ui->lblStatusVal->setText(QStringLiteral("Error: %1").arg(res.errorText));
        return;
    }

    m_lastOutputs = res.outputs;
    m_lastMetrics = res.finalMetrics;

    updateViewers();
    updateMetricsDisplay(res);
    ui->lblStatusVal->setText(QStringLiteral("Done"));

    emit processingFinished(m_lastOutputs, m_lastMetrics);
}

void N5Form::updateViewers()
{
    if (m_lastOutputs.input8U.empty() || m_lastOutputs.output8U.empty()) {
        return;
    }

    ui->canvasInput->setImage(matToQImage(m_lastOutputs.input8U));
    ui->canvasOutput->setImage(matToQImage(m_lastOutputs.output8U));

    auto toDisp8U = [](const cv::Mat& f64) -> cv::Mat {
        cv::Mat clamped, u8;
        cv::min(cv::max(f64, 0.0), 1.0, clamped);
        clamped.convertTo(u8, CV_8U, 255.0);
        return u8;
    };

    ui->canvasB1->setImage(matToQImage(toDisp8U(m_lastOutputs.B1)));
    ui->canvasR1->setImage(matToQImage(msgf::N5Algorithm::normalizeForDisplay(m_lastOutputs.R1)));
    ui->canvasC->setImage(matToQImage(m_lastOutputs.C_clahe8U));
    ui->canvasU->setImage(matToQImage(m_lastOutputs.U_stage18U));
    ui->canvasB2->setImage(matToQImage(toDisp8U(m_lastOutputs.B2)));
    ui->canvasR2->setImage(matToQImage(msgf::N5Algorithm::normalizeForDisplay(m_lastOutputs.R2)));
}

void N5Form::updateMetricsDisplay(const WorkerResult& res)
{
    if (!res.finalMetrics.valid) {
        return;
    }

    const auto& in = res.finalMetrics.inputStats;
    const auto& out = res.finalMetrics.outputStats;

    ui->lblPsnrVal->setText(QStringLiteral("%1 dB").arg(res.finalMetrics.psnr, 0, 'f', 2));
    ui->lblIeVal->setText(QStringLiteral("%1 → %2 (%3)")
                              .arg(in.ie, 0, 'f', 2)
                              .arg(out.ie, 0, 'f', 2)
                              .arg(formatDeltaN5(out.ie - in.ie, 2, false)));
    ui->lblAgVal->setText(QStringLiteral("%1 → %2 (%3)")
                              .arg(in.ag, 0, 'f', 2)
                              .arg(out.ag, 0, 'f', 2)
                              .arg(formatDeltaN5(out.ag - in.ag, 2, false)));
    ui->lblPiqeVal->setText(QStringLiteral("%1 → %2 (%3)")
                                .arg(in.piqe, 0, 'f', 2)
                                .arg(out.piqe, 0, 'f', 2)
                                .arg(formatDeltaN5(out.piqe - in.piqe, 2, true)));
    ui->lblRegVal->setText(QStringLiteral("%1 / %2 (const=%3)")
                               .arg(res.outputs.baseRegularization, 0, 'e', 2)
                               .arg(res.outputs.smallRegularization, 0, 'e', 2)
                               .arg(res.outputs.constantPreserved ? QStringLiteral("true") : QStringLiteral("false")));
    ui->lblRuntimeVal->setText(QStringLiteral("%1 ms").arg(res.outputs.elapsedMs, 0, 'f', 2));

    // Update 4-row Stage-by-Stage Ablation table
    auto setRow = [&](int row, const QString& psnrStr, const msgf::MsgfPlaneStats& st) {
        ui->tblAblation->item(row, 1)->setText(psnrStr);
        ui->tblAblation->item(row, 2)->setText(QString::number(st.ie, 'f', 2));
        ui->tblAblation->item(row, 3)->setText(QString::number(st.ag, 'f', 2));
        ui->tblAblation->item(row, 4)->setText(QString::number(st.piqe, 'f', 2));
    };
    setRow(0, QStringLiteral("—"), in);
    setRow(1, QString::number(res.claheMetrics.psnr, 'f', 2), res.claheMetrics.outputStats);
    setRow(2, QString::number(res.stage1Metrics.psnr, 'f', 2), res.stage1Metrics.outputStats);
    setRow(3, QString::number(res.finalMetrics.psnr, 'f', 2), out);
}

void N5Form::onSaveOutput()
{
    if (m_lastOutputs.output8U.empty()) {
        QMessageBox::information(this, QStringLiteral("Save Output"),
                                 QStringLiteral("No processed output image available yet."));
        return;
    }

    const QString outPath = QFileDialog::getSaveFileName(
        this,
        QStringLiteral("Save Enhanced N5 Output"),
        QDir::homePath() + QStringLiteral("/n5_output.png"),
        QStringLiteral("PNG Image (*.png);;JPEG Image (*.jpg);;TIFF Image (*.tif)")
    );
    if (!outPath.isEmpty() && !cv::imwrite(outPath.toStdString(), m_lastOutputs.output8U)) {
        QMessageBox::warning(this, QStringLiteral("Save Output"),
                             QStringLiteral("Failed to write file:\n%1").arg(outPath));
    }
}

void N5Form::onExportStages()
{
    if (m_lastOutputs.output8U.empty()) {
        QMessageBox::information(this, QStringLiteral("Export Layers"),
                                 QStringLiteral("No processed output available yet."));
        return;
    }

    const QString dirPath = QFileDialog::getExistingDirectory(
        this,
        QStringLiteral("Select Folder to Export All N5 Stages"),
        QDir::homePath()
    );
    if (dirPath.isEmpty()) {
        return;
    }

    const QDir dir(dirPath);
    auto save64 = [&](const QString& name, const cv::Mat& mat) -> bool {
        cv::Mat clamped, u8;
        cv::min(cv::max(mat, 0.0), 1.0, clamped);
        clamped.convertTo(u8, CV_8U, 255.0);
        return cv::imwrite(dir.filePath(name).toStdString(), u8);
    };

    const bool ok =
        cv::imwrite(dir.filePath(QStringLiteral("01_input8U.png")).toStdString(), m_lastOutputs.input8U)
        && save64(QStringLiteral("02_B1_stage1_base.png"), m_lastOutputs.B1)
        && cv::imwrite(dir.filePath(QStringLiteral("03_R1_stage1_residual.png")).toStdString(),
                       msgf::N5Algorithm::normalizeForDisplay(m_lastOutputs.R1))
        && cv::imwrite(dir.filePath(QStringLiteral("04_C_clahe_base.png")).toStdString(), m_lastOutputs.C_clahe8U)
        && cv::imwrite(dir.filePath(QStringLiteral("05_U_stage1_quantized.png")).toStdString(), m_lastOutputs.U_stage18U)
        && save64(QStringLiteral("06_B2_fine_base.png"), m_lastOutputs.B2)
        && cv::imwrite(dir.filePath(QStringLiteral("07_R2_fine_residual.png")).toStdString(),
                       msgf::N5Algorithm::normalizeForDisplay(m_lastOutputs.R2))
        && cv::imwrite(dir.filePath(QStringLiteral("08_Y_n5_output.png")).toStdString(), m_lastOutputs.output8U);

    if (!ok) {
        QMessageBox::warning(this, QStringLiteral("Export Layers"),
                             QStringLiteral("Some stages could not be written to:\n%1").arg(dirPath));
    } else {
        QMessageBox::information(this, QStringLiteral("Export Layers"),
                                 QStringLiteral("Exported all 8 N5 stage images to:\n%1").arg(dirPath));
    }
}

void N5Form::onSyncViewChanged(double scale, const QPointF& offset)
{
    if (m_isSyncingViews) {
        return;
    }
    m_isSyncingViews = true;
    auto* source = qobject_cast<ImageCanvas*>(sender());
    for (ImageCanvas* canvas : allCanvases()) {
        if (canvas != source) {
            canvas->setZoomAndPan(scale, offset);
        }
    }
    m_isSyncingViews = false;
}

void N5Form::onSyncViewReset()
{
    if (m_isSyncingViews) {
        return;
    }
    m_isSyncingViews = true;
    auto* source = qobject_cast<ImageCanvas*>(sender());
    for (ImageCanvas* canvas : allCanvases()) {
        if (canvas != source) {
            canvas->resetView();
        }
    }
    m_isSyncingViews = false;
}

void N5Form::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Left) {
        onPrevImage();
        event->accept();
        return;
    }
    if (event->key() == Qt::Key_Right) {
        onNextImage();
        event->accept();
        return;
    }
    QWidget::keyPressEvent(event);
}

QImage N5Form::matToQImage(const cv::Mat& mat)
{
    if (mat.empty()) {
        return {};
    }
    if (mat.type() == CV_8UC1) {
        QImage img(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Grayscale8);
        return img.copy();
    }
    if (mat.type() == CV_8UC3) {
        QImage img(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_BGR888);
        return img.copy();
    }
    cv::Mat gray = MsgfAlgorithm::toGray8U(mat);
    QImage img(gray.data, gray.cols, gray.rows, static_cast<int>(gray.step), QImage::Format_Grayscale8);
    return img.copy();
}
