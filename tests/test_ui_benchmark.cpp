#include "ui/DatasetBenchmarkWidget.h"
#include "ui/MsgfForm.h"
#include "ui/NumericTableItem.h"

#include <QApplication>
#include <QComboBox>
#include <QDir>
#include <QFileInfo>
#include <QLineEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QTabWidget>
#include <QTableWidget>
#include <QtCharts/QChartView>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {

void require(bool cond, const std::string& msg)
{
    if (!cond) {
        throw std::runtime_error("UI test failed: " + msg);
    }
}

QString findSampleDir()
{
    namespace fs = std::filesystem;
    for (const char* cand : {"sample_images", "../sample_images", "../../sample_images"}) {
        if (fs::exists(fs::path(cand) / "scene1_original.png")) {
            return QString::fromStdString(fs::canonical(cand).string());
        }
    }
    return {};
}

} // namespace

int main(int argc, char* argv[])
{
    qputenv("QT_QPA_PLATFORM", QByteArray("offscreen"));
    QApplication app(argc, argv);

    try {
        // 1. NumericTableItem must sort numerically (9.5 < 12.3, unlike lexicographic "12.3" < "9.5")
        NumericTableItem itemSmall(9.5, 2);
        NumericTableItem itemLarge(12.3, 2);
        require(itemSmall < itemLarge, "NumericTableItem numeric sort failed (9.5 vs 12.3)");
        require(!(itemLarge < itemSmall), "NumericTableItem anti-symmetry failed");

        NumericTableItem meanStdA(6.47, QStringLiteral("6.47 ± 0.31"));
        NumericTableItem meanStdB(12.10, QStringLiteral("12.10 ± 1.05"));
        require(meanStdA < meanStdB, "NumericTableItem Mean+/-Std sort failed");

        // 2. Instantiate MsgfForm (4 tabs including Dataset Benchmark & Charts)
        MsgfForm form;
        form.resize(1360, 840);
        form.show();
        QApplication::processEvents();

        auto* tabs = form.findChild<QTabWidget*>(QStringLiteral("viewerTabs"));
        require(tabs != nullptr && tabs->count() == 4, "MsgfForm should have 4 viewer tabs");

        auto* bench = form.findChild<DatasetBenchmarkWidget*>(QStringLiteral("benchmarkWidget"));
        require(bench != nullptr, "DatasetBenchmarkWidget missing inside MsgfForm");

        const QString sampleDir = findSampleDir();
        if (!sampleDir.isEmpty()) {
            bench->setDatasetDirectory(sampleDir);

            // Switch to Benchmark tab and click Run
            tabs->setCurrentIndex(3);
            auto* runBtn = bench->findChild<QPushButton*>(QStringLiteral("btnRunBench"));
            require(runBtn != nullptr, "Run Benchmark button not found");
            runBtn->click();

            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
            while (bench->isRunning() && std::chrono::steady_clock::now() < deadline) {
                QApplication::processEvents();
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            QApplication::processEvents();
            require(!bench->isRunning(), "Benchmark did not finish within timeout");

            auto* summaryTbl = bench->findChild<QTableWidget*>(QStringLiteral("tblSummary"));
            auto* detailTbl = bench->findChild<QTableWidget*>(QStringLiteral("tblDetail"));
            require(summaryTbl != nullptr && detailTbl != nullptr, "Expected Summary & Detail tables");

            // Default checked: Original + HE + GF&DDE + MSGF-TIR => 4 summary rows
            require(summaryTbl->rowCount() == 4, "Expected 4 rows in summary table");
            require(detailTbl->rowCount() == 3, "Expected 3 sample scene rows in detail table");

            // Test switching all 3 QtChart modes + metrics and grabbing rendered chart
            auto* chartTypeCombo = bench->findChild<QComboBox*>(QStringLiteral("comboChartType"));
            auto* chartMetricCombo = bench->findChild<QComboBox*>(QStringLiteral("comboChartMetric"));
            auto* chartView = bench->findChild<QChartView*>(QStringLiteral("benchChartView"));
            require(chartTypeCombo != nullptr && chartMetricCombo != nullptr && chartView != nullptr,
                    "Missing chart widgets");

            for (int t = 0; t < chartTypeCombo->count(); ++t) {
                chartTypeCombo->setCurrentIndex(t);
                for (int m = 0; m < chartMetricCombo->count(); ++m) {
                    chartMetricCombo->setCurrentIndex(m);
                    QApplication::processEvents();
                }
            }
            const QPixmap chartPix = chartView->grab();
            require(!chartPix.isNull() && chartPix.width() > 50, "Chart grab failed");

            // Test double-click 2-way navigation from Detail table back to Compare tab (index 0)
            emit detailTbl->cellDoubleClicked(0, 1);
            QApplication::processEvents();
            require(tabs->currentIndex() == 0, "Double-click should switch back to Compare tab");
        }

        std::cout << "All MSGF-TIR UI, Chart & Benchmark integration tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
