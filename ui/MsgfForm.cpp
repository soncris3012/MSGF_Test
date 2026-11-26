#include "ui/MsgfForm.h"
#include "ui_MsgfForm.h"

#include "ui/DatasetBenchmarkWidget.h"
#include "ui/ImageCanvas.h"

#include <QApplication>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHeaderView>
#include <QMessageBox>
#include <QSignalBlocker>
#include <QTableWidgetItem>
#include <QtConcurrent/QtConcurrentRun>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <exception>

namespace {

QString formatDelta(double delta, int precision = 2, bool lowerIsBetter = false)
{
    const QString sign = (delta >= 0.0) ? QStringLiteral("+") : QStringLiteral("");
    const QString arrow = (lowerIsBetter ? (delta <= 0.0 ? QStringLiteral("✓") : QStringLiteral(""))
                                         : (delta >= 0.0 ? QStringLiteral("✓") : QStringLiteral("")));
    return QStringLiteral("%1%2 %3").arg(sign).arg(delta, 0, 'f', precision).arg(arrow).trimmed();
}

} // namespace

MsgfForm::MsgfForm(QWidget* parent)
    : QWidget(parent)
    , ui(std::make_unique<Ui::MsgfForm>())
    , m_debounceTimer(new QTimer(this))
    , m_watcher(new QFutureWatcher<WorkerResult>(this))
{
    ui->setupUi(this);

    m_debounceTimer->setSingleShot(true);
    m_debounceTimer->setInterval(60);

    applyTheme();
    populateBenchmarkTable();
    setupConnections();

    // Set sensible initial splitter proportions (~340 | ~790 | ~350)
    ui->mainSplitter->setSizes({340, 790, 350});

    // Initialize canvas badges
    ui->canvasInput->setTitle(QStringLiteral("Original Input (I)"));
    ui->canvasOutput->setTitle(QStringLiteral("MSGF-TIR Output"));
    ui->canvasDecompOrig->setTitle(QStringLiteral("(a) Original Input I"));
    ui->canvasDecompD1->setTitle(QStringLiteral("(b) Small Detail D1 (norm)"));
    ui->canvasDecompD2->setTitle(QStringLiteral("(c) Medium Detail D2 (norm)"));
    ui->canvasDecompD3->setTitle(QStringLiteral("(d) Large Detail D3 (norm)"));
    ui->canvasDecompB3->setTitle(QStringLiteral("(e) Base Layer B3"));
    ui->canvasDecompIB->setTitle(QStringLiteral("(f) Enhanced Base IB (CLAHE)"));
    ui->canvasIDFused->setTitle(QStringLiteral("Fused Details ID (norm)"));
    ui->canvasPaperRef->setTitle(QStringLiteral("Paper Reference (Fig. 6-8e)"));
    ui->canvasPaperRef->setPlaceholderText(QStringLiteral("No paper reference matched for this file"));

    initializeSampleDirectory();
}

MsgfForm::~MsgfForm()
{
    if (m_debounceTimer->isActive()) {
        m_debounceTimer->stop();
    }
    if (m_watcher->isRunning()) {
        m_watcher->waitForFinished();
    }
}

void MsgfForm::applyTheme()
{
    setStyleSheet(QStringLiteral(R"(
        QWidget#MsgfForm {
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
            color: #8ab4f8;
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
            border-color: #5b8af0;
        }
        QPushButton:pressed {
            background-color: #252830;
        }
        QPushButton#btnRun {
            background-color: #2857a4;
            border-color: #3b74d1;
            font-weight: 600;
        }
        QPushButton#btnRun:hover {
            background-color: #3268c0;
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
            border-bottom: 2px solid #5b8af0;
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

void MsgfForm::populateBenchmarkTable()
{
    const auto& rows = msgf::paperTable1Benchmarks();
    auto* tbl = ui->tblBenchmark;
    tbl->setColumnCount(5);
    tbl->setHorizontalHeaderLabels({
        QStringLiteral("Scene / Method"),
        QStringLiteral("PSNR"),
        QStringLiteral("IE"),
        QStringLiteral("AG"),
        QStringLiteral("PIQE")
    });
    tbl->setRowCount(static_cast<int>(rows.size()));
    tbl->verticalHeader()->setVisible(false);

    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
        const auto& r = rows[static_cast<std::size_t>(i)];
        const QString label = QStringLiteral("%1 · %2")
            .arg(QString::fromStdString(r.sceneName))
            .arg(QString::fromStdString(r.algorithm));

        auto* c0 = new QTableWidgetItem(label);
        auto* c1 = new QTableWidgetItem(QString::number(r.psnr, 'f', 2));
        auto* c2 = new QTableWidgetItem(QString::number(r.ie, 'f', 2));
        auto* c3 = new QTableWidgetItem(QString::number(r.ag, 'f', 2));
        auto* c4 = new QTableWidgetItem(QString::number(r.piqe, 'f', 2));

        c1->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
        c2->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
        c3->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
        c4->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);

        if (r.isProposed) {
            QFont bold = c0->font();
            bold.setBold(true);
            for (auto* item : {c0, c1, c2, c3, c4}) {
                item->setFont(bold);
                item->setBackground(QColor(36, 62, 102));
                item->setForeground(QColor(240, 246, 255));
            }
        }

        tbl->setItem(i, 0, c0);
        tbl->setItem(i, 1, c1);
        tbl->setItem(i, 2, c2);
        tbl->setItem(i, 3, c3);
        tbl->setItem(i, 4, c4);
        tbl->setRowHeight(i, 22);
    }

    auto* hh = tbl->horizontalHeader();
    hh->setSectionResizeMode(0, QHeaderView::Stretch);
    for (int col = 1; col < 5; ++col) {
        hh->setSectionResizeMode(col, QHeaderView::ResizeToContents);
    }
}

void MsgfForm::setupConnections()
{
    connect(ui->btnLoadImage, &QPushButton::clicked, this, &MsgfForm::onLoadImage);
    connect(ui->btnOpenFolder, &QPushButton::clicked, this, &MsgfForm::onOpenFolder);
    connect(ui->comboScene, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MsgfForm::onSceneSelected);
    connect(ui->btnPrev, &QPushButton::clicked, this, &MsgfForm::onPrevImage);
    connect(ui->btnNext, &QPushButton::clicked, this, &MsgfForm::onNextImage);

    connect(ui->btnRun, &QPushButton::clicked, this, &MsgfForm::onRunClicked);
    connect(ui->btnReset, &QPushButton::clicked, this, &MsgfForm::onResetDefaults);
    connect(ui->btnSaveOutput, &QPushButton::clicked, this, &MsgfForm::onSaveOutput);
    connect(ui->btnExportStages, &QPushButton::clicked, this, &MsgfForm::onExportStages);

    for (QSpinBox* sb : {ui->spinR1, ui->spinR2, ui->spinR3, ui->spinTileGrid}) {
        connect(sb, QOverload<int>::of(&QSpinBox::valueChanged),
                this, &MsgfForm::onParamEdited);
    }
    for (QDoubleSpinBox* dsb : {
             ui->spinEps1, ui->spinEps2, ui->spinEps3,
             ui->spinClipLimit,
             ui->spinAlpha1, ui->spinAlpha2, ui->spinAlpha3,
             ui->spinW1, ui->spinW2, ui->spinW3,
             ui->spinP, ui->spinQ
         }) {
        connect(dsb, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this, &MsgfForm::onParamEdited);
    }

    connect(m_debounceTimer, &QTimer::timeout, this, &MsgfForm::startBackgroundRun);
    connect(m_watcher, &QFutureWatcher<WorkerResult>::finished,
            this, &MsgfForm::onWorkerFinished);

    // Synchronize zoom/pan across all ImageCanvas panels
    for (ImageCanvas* canvas : allCanvases()) {
        connect(canvas, &ImageCanvas::viewChanged, this, &MsgfForm::onSyncViewChanged);
        connect(canvas, &ImageCanvas::viewReset, this, &MsgfForm::onSyncViewReset);
    }

    // Two-way link: double-clicking an image row in Dataset Benchmark opens it in Compare
    connect(ui->benchmarkWidget, &DatasetBenchmarkWidget::imageActivated,
            this, [this](const QString& filePath) {
                if (loadImageFile(filePath)) {
                    ui->viewerTabs->setCurrentIndex(0);
                }
            });
}

std::vector<ImageCanvas*> MsgfForm::allCanvases() const
{
    return {
        ui->canvasInput,
        ui->canvasOutput,
        ui->canvasDecompOrig,
        ui->canvasDecompD1,
        ui->canvasDecompD2,
        ui->canvasDecompD3,
        ui->canvasDecompB3,
        ui->canvasDecompIB,
        ui->canvasIDFused,
        ui->canvasPaperRef
    };
}

void MsgfForm::initializeSampleDirectory()
{
    ui->benchmarkWidget->setMsgfParams(currentParams());
    m_sampleDir = locateSampleDirectory();
    if (!m_sampleDir.isEmpty()) {
        loadFolderImages(m_sampleDir, true);
        ui->benchmarkWidget->setDatasetDirectory(m_sampleDir);
    } else {
        updateNavigationButtons();
    }
}

QString MsgfForm::locateSampleDirectory()
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

void MsgfForm::loadFolderImages(const QString& dirPath, bool isSampleDir)
{
    const QDir dir(dirPath);
    if (!dir.exists()) {
        return;
    }

    const QStringList filters = {
        QStringLiteral("*.png"), QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"),
        QStringLiteral("*.bmp"), QStringLiteral("*.tif"), QStringLiteral("*.tiff")
    };
    const QFileInfoList infos = dir.entryInfoList(filters, QDir::Files, QDir::Name);

    m_imageFiles.clear();
    {
        const QSignalBlocker blocker(ui->comboScene);
        ui->comboScene->clear();

        for (const QFileInfo& fi : infos) {
            const QString fname = fi.fileName();
            if (isSampleDir) {
                // Show only the 3 original paper test scenes in the quick scene selector
                if (!fname.endsWith(QStringLiteral("_original.png"))) {
                    continue;
                }
            }
            m_imageFiles.push_back(fi.absoluteFilePath());
            QString displayName = fname;
            if (fname == QStringLiteral("scene1_original.png")) {
                displayName = QStringLiteral("Scene 1 (iRay T3-Pro)");
            } else if (fname == QStringLiteral("scene2_original.png")) {
                displayName = QStringLiteral("Scene 2 (FLIR Vehicle)");
            } else if (fname == QStringLiteral("scene3_original.png")) {
                displayName = QStringLiteral("Scene 3 (FLIR Pedestrian)");
            }
            ui->comboScene->addItem(displayName, fi.absoluteFilePath());
        }
    }

    m_currentFileIndex = m_imageFiles.empty() ? -1 : 0;
    updateNavigationButtons();
    if (m_currentFileIndex >= 0) {
        const QSignalBlocker blocker(ui->comboScene);
        ui->comboScene->setCurrentIndex(0);
        loadImageFile(m_imageFiles[0]);
    }
}

void MsgfForm::updateNavigationButtons()
{
    const bool hasMultiple = m_imageFiles.size() > 1;
    ui->btnPrev->setEnabled(hasMultiple && m_currentFileIndex > 0);
    ui->btnNext->setEnabled(hasMultiple && m_currentFileIndex >= 0
                            && m_currentFileIndex + 1 < static_cast<int>(m_imageFiles.size()));
}

MsgfParams MsgfForm::currentParams() const
{
    MsgfParams p;
    p.r1 = ui->spinR1->value();
    p.eps1 = ui->spinEps1->value();
    p.r2 = ui->spinR2->value();
    p.eps2 = ui->spinEps2->value();
    p.r3 = ui->spinR3->value();
    p.eps3 = ui->spinEps3->value();

    p.claheClipLimit = ui->spinClipLimit->value();
    p.claheTileGrid = ui->spinTileGrid->value();

    p.alpha1 = ui->spinAlpha1->value();
    p.alpha2 = ui->spinAlpha2->value();
    p.alpha3 = ui->spinAlpha3->value();

    p.w1 = ui->spinW1->value();
    p.w2 = ui->spinW2->value();
    p.w3 = ui->spinW3->value();

    p.p = ui->spinP->value();
    p.q = ui->spinQ->value();
    return p;
}

void MsgfForm::setParams(const MsgfParams& p)
{
    m_isUpdatingControls = true;
    ui->spinR1->setValue(p.r1);
    ui->spinEps1->setValue(p.eps1);
    ui->spinR2->setValue(p.r2);
    ui->spinEps2->setValue(p.eps2);
    ui->spinR3->setValue(p.r3);
    ui->spinEps3->setValue(p.eps3);

    ui->spinClipLimit->setValue(p.claheClipLimit);
    ui->spinTileGrid->setValue(p.claheTileGrid);

    ui->spinAlpha1->setValue(p.alpha1);
    ui->spinAlpha2->setValue(p.alpha2);
    ui->spinAlpha3->setValue(p.alpha3);

    ui->spinW1->setValue(p.w1);
    ui->spinW2->setValue(p.w2);
    ui->spinW3->setValue(p.w3);

    ui->spinP->setValue(p.p);
    ui->spinQ->setValue(p.q);
    m_isUpdatingControls = false;

    ui->benchmarkWidget->setMsgfParams(p);
    scheduleProcessing(true);
}

void MsgfForm::setInputImage(const cv::Mat& image, const QString& label)
{
    if (image.empty()) {
        return;
    }
    m_inputMat = image.clone();
    m_paperRefMat.release();
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

bool MsgfForm::loadImageFile(const QString& filePath)
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

    // Check if matching *_paper_ours.png exists alongside *_original.png
    m_paperRefMat.release();
    if (fi.fileName().endsWith(QStringLiteral("_original.png"))) {
        QString refName = fi.fileName();
        refName.replace(QStringLiteral("_original.png"), QStringLiteral("_paper_ours.png"));
        const QString refPath = fi.dir().filePath(refName);
        if (QFileInfo::exists(refPath)) {
            m_paperRefMat = cv::imread(refPath.toStdString(), cv::IMREAD_UNCHANGED);
        }
    }

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

void MsgfForm::onLoadImage()
{
    const QString startDir = m_sampleDir.isEmpty() ? QDir::homePath() : m_sampleDir;
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

void MsgfForm::onOpenFolder()
{
    const QString startDir = m_sampleDir.isEmpty() ? QDir::homePath() : m_sampleDir;
    const QString dirPath = QFileDialog::getExistingDirectory(
        this,
        QStringLiteral("Select Folder of Thermal Images"),
        startDir
    );
    if (!dirPath.isEmpty()) {
        loadFolderImages(dirPath, false);
        if (!ui->benchmarkWidget->isRunning()) {
            ui->benchmarkWidget->setDatasetDirectory(dirPath);
        }
    }
}

void MsgfForm::onSceneSelected(int index)
{
    if (index < 0 || index >= static_cast<int>(m_imageFiles.size())) {
        return;
    }
    m_currentFileIndex = index;
    updateNavigationButtons();
    loadImageFile(m_imageFiles[static_cast<std::size_t>(index)]);
}

void MsgfForm::onPrevImage()
{
    if (m_currentFileIndex > 0) {
        ui->comboScene->setCurrentIndex(m_currentFileIndex - 1);
    }
}

void MsgfForm::onNextImage()
{
    if (m_currentFileIndex >= 0
        && m_currentFileIndex + 1 < static_cast<int>(m_imageFiles.size())) {
        ui->comboScene->setCurrentIndex(m_currentFileIndex + 1);
    }
}

void MsgfForm::onParamEdited()
{
    if (m_isUpdatingControls) {
        return;
    }
    ui->benchmarkWidget->setMsgfParams(currentParams());
    if (ui->chkAutoUpdate->isChecked()) {
        scheduleProcessing(false);
    }
}

void MsgfForm::onRunClicked()
{
    scheduleProcessing(true);
}

void MsgfForm::onResetDefaults()
{
    setParams(MsgfParams{});
}

void MsgfForm::scheduleProcessing(bool immediate)
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

void MsgfForm::startBackgroundRun()
{
    if (m_inputMat.empty()) {
        return;
    }

    const std::uint64_t gen = ++m_generation;
    const cv::Mat imgCopy = m_inputMat.clone();
    const MsgfParams params = currentParams();

    ui->lblStatusVal->setText(QStringLiteral("Processing..."));

    auto future = QtConcurrent::run([imgCopy, params, gen]() -> WorkerResult {
        WorkerResult res;
        res.generation = gen;
        try {
            res.outputs = MsgfAlgorithm::run(imgCopy, params);
            res.metrics = MsgfMetrics::evaluate(res.outputs.input8U, res.outputs.output8U);
            res.ok = true;
        } catch (const std::exception& e) {
            res.ok = false;
            res.errorText = QString::fromUtf8(e.what());
        }
        return res;
    });
    m_watcher->setFuture(future);
}

void MsgfForm::onWorkerFinished()
{
    const WorkerResult res = m_watcher->result();
    if (res.generation != m_generation) {
        // Newer parameter/image update is already queued or running; discard stale frame.
        return;
    }

    if (!res.ok) {
        ui->lblStatusVal->setText(QStringLiteral("Error: %1").arg(res.errorText));
        return;
    }

    m_lastOutputs = res.outputs;
    m_lastMetrics = res.metrics;

    updateViewers();
    updateMetricsDisplay();
    ui->lblStatusVal->setText(QStringLiteral("Done"));

    emit processingFinished(m_lastOutputs, m_lastMetrics);
}

void MsgfForm::updateViewers()
{
    if (m_lastOutputs.input8U.empty() || m_lastOutputs.output8U.empty()) {
        return;
    }

    const QImage imgIn = matToQImage(m_lastOutputs.input8U);
    const QImage imgOut = matToQImage(m_lastOutputs.output8U);

    ui->canvasInput->setImage(imgIn);
    ui->canvasOutput->setImage(imgOut);

    // Tab 2: Figure 3 decomposition (D1..D3 normalized for signed display + B3 + IB)
    ui->canvasDecompOrig->setImage(imgIn);
    ui->canvasDecompD1->setImage(matToQImage(MsgfAlgorithm::normalizeForDisplay(m_lastOutputs.D1)));
    ui->canvasDecompD2->setImage(matToQImage(MsgfAlgorithm::normalizeForDisplay(m_lastOutputs.D2)));
    ui->canvasDecompD3->setImage(matToQImage(MsgfAlgorithm::normalizeForDisplay(m_lastOutputs.D3)));

    cv::Mat b3Disp;
    cv::min(cv::max(m_lastOutputs.B3, 0.0f), 1.0f, b3Disp);
    b3Disp.convertTo(b3Disp, CV_8U, 255.0);
    ui->canvasDecompB3->setImage(matToQImage(b3Disp));

    cv::Mat ibDisp;
    cv::min(cv::max(m_lastOutputs.IB_clahe, 0.0f), 1.0f, ibDisp);
    ibDisp.convertTo(ibDisp, CV_8U, 255.0);
    ui->canvasDecompIB->setImage(matToQImage(ibDisp));

    // Tab 3: Fused details & optional paper reference
    ui->canvasIDFused->setImage(matToQImage(MsgfAlgorithm::normalizeForDisplay(m_lastOutputs.ID_fused)));
    if (!m_paperRefMat.empty()) {
        ui->canvasPaperRef->setImage(matToQImage(MsgfAlgorithm::toGray8U(m_paperRefMat)));
    } else {
        ui->canvasPaperRef->setImage(QImage{});
    }
}

void MsgfForm::updateMetricsDisplay()
{
    if (!m_lastMetrics.valid) {
        return;
    }

    const auto& in = m_lastMetrics.inputStats;
    const auto& out = m_lastMetrics.outputStats;

    ui->lblPsnrVal->setText(QStringLiteral("%1 dB").arg(m_lastMetrics.psnr, 0, 'f', 2));
    ui->lblIeVal->setText(QStringLiteral("%1 → %2 (%3)")
                              .arg(in.ie, 0, 'f', 2)
                              .arg(out.ie, 0, 'f', 2)
                              .arg(formatDelta(out.ie - in.ie, 2, false)));
    ui->lblAgVal->setText(QStringLiteral("%1 → %2 (%3)")
                              .arg(in.ag, 0, 'f', 2)
                              .arg(out.ag, 0, 'f', 2)
                              .arg(formatDelta(out.ag - in.ag, 2, false)));
    ui->lblPiqeVal->setText(QStringLiteral("%1 → %2 (%3)")
                                .arg(in.piqe, 0, 'f', 2)
                                .arg(out.piqe, 0, 'f', 2)
                                .arg(formatDelta(out.piqe - in.piqe, 2, true)));
    ui->lblRuntimeVal->setText(QStringLiteral("%1 ms").arg(m_lastOutputs.elapsedMs, 0, 'f', 2));
}

void MsgfForm::onSaveOutput()
{
    if (m_lastOutputs.output8U.empty()) {
        QMessageBox::information(this, QStringLiteral("Save Output"),
                                 QStringLiteral("No processed output image available yet."));
        return;
    }

    const QString defaultName = QDir::homePath() + QStringLiteral("/msgf_output.png");
    const QString outPath = QFileDialog::getSaveFileName(
        this,
        QStringLiteral("Save Enhanced Output"),
        defaultName,
        QStringLiteral("PNG Image (*.png);;JPEG Image (*.jpg);;TIFF Image (*.tif)")
    );
    if (outPath.isEmpty()) {
        return;
    }

    if (!cv::imwrite(outPath.toStdString(), m_lastOutputs.output8U)) {
        QMessageBox::warning(this, QStringLiteral("Save Output"),
                             QStringLiteral("Failed to write file:\n%1").arg(outPath));
    }
}

void MsgfForm::onExportStages()
{
    if (m_lastOutputs.output8U.empty()) {
        QMessageBox::information(this, QStringLiteral("Export Layers"),
                                 QStringLiteral("No processed output available yet."));
        return;
    }

    const QString dirPath = QFileDialog::getExistingDirectory(
        this,
        QStringLiteral("Select Folder to Export All MSGF-TIR Stages"),
        QDir::homePath()
    );
    if (dirPath.isEmpty()) {
        return;
    }

    const QDir dir(dirPath);
    auto saveFloat01 = [&](const QString& name, const cv::Mat& mat) -> bool {
        cv::Mat clamped;
        cv::min(cv::max(mat, 0.0f), 1.0f, clamped);
        cv::Mat u8;
        clamped.convertTo(u8, CV_8U, 255.0);
        return cv::imwrite(dir.filePath(name).toStdString(), u8);
    };

    const bool ok =
        cv::imwrite(dir.filePath(QStringLiteral("01_input8U.png")).toStdString(), m_lastOutputs.input8U)
        && saveFloat01(QStringLiteral("02_B1_small_base.png"), m_lastOutputs.B1)
        && saveFloat01(QStringLiteral("03_B2_medium_base.png"), m_lastOutputs.B2)
        && saveFloat01(QStringLiteral("04_B3_large_base.png"), m_lastOutputs.B3)
        && cv::imwrite(dir.filePath(QStringLiteral("05_D1_small_detail.png")).toStdString(),
                       MsgfAlgorithm::normalizeForDisplay(m_lastOutputs.D1))
        && cv::imwrite(dir.filePath(QStringLiteral("06_D2_medium_detail.png")).toStdString(),
                       MsgfAlgorithm::normalizeForDisplay(m_lastOutputs.D2))
        && cv::imwrite(dir.filePath(QStringLiteral("07_D3_large_detail.png")).toStdString(),
                       MsgfAlgorithm::normalizeForDisplay(m_lastOutputs.D3))
        && saveFloat01(QStringLiteral("08_IB_clahe_base.png"), m_lastOutputs.IB_clahe)
        && cv::imwrite(dir.filePath(QStringLiteral("09_ID_fused_details.png")).toStdString(),
                       MsgfAlgorithm::normalizeForDisplay(m_lastOutputs.ID_fused))
        && cv::imwrite(dir.filePath(QStringLiteral("10_output_fused.png")).toStdString(), m_lastOutputs.output8U);

    if (!ok) {
        QMessageBox::warning(this, QStringLiteral("Export Layers"),
                             QStringLiteral("Some layers could not be written to:\n%1").arg(dirPath));
    } else {
        QMessageBox::information(this, QStringLiteral("Export Layers"),
                                 QStringLiteral("Exported all 10 stage images to:\n%1").arg(dirPath));
    }
}

void MsgfForm::onSyncViewChanged(double scale, const QPointF& offset)
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

void MsgfForm::onSyncViewReset()
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

void MsgfForm::keyPressEvent(QKeyEvent* event)
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

QImage MsgfForm::matToQImage(const cv::Mat& mat)
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
