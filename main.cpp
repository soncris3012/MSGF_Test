#include "ui/MsgfForm.h"
#include "ui/N5Form.h"

#include <QApplication>
#include <QMainWindow>
#include <QTabWidget>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("MSGF-TIR & N5"));
    app.setOrganizationName(QStringLiteral("MSGF_Test"));

    QMainWindow mainWindow;
    mainWindow.setWindowTitle(
        QStringLiteral("Thermal IR Enhancement Workbench — MSGF-TIR & Accepted N5")
    );

    auto* pageTabs = new QTabWidget(&mainWindow);
    pageTabs->setDocumentMode(true);
    pageTabs->setStyleSheet(QStringLiteral(R"(
        QTabWidget::pane { border: none; background-color: #1e1f22; }
        QTabBar::tab {
            background-color: #27292e;
            color: #b8bec9;
            padding: 8px 18px;
            font-weight: 600;
            border: 1px solid #383a40;
            border-bottom: none;
            margin-right: 3px;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
        }
        QTabBar::tab:selected {
            background-color: #1e1f22;
            color: #ffffff;
            border-bottom: 2px solid #5b8af0;
        }
    )"));

    auto* msgfForm = new MsgfForm(pageTabs);
    auto* n5Form = new N5Form(pageTabs);

    pageTabs->addTab(msgfForm, QStringLiteral("MSGF-TIR (3-Scale Guided Filter)"));
    pageTabs->addTab(n5Form, QStringLiteral("N5 (Guided-CLAHE + Post-Half)"));

    mainWindow.setCentralWidget(pageTabs);
    mainWindow.resize(1500, 900);

    if (argc > 1) {
        const QString imgPath = QString::fromLocal8Bit(argv[1]);
        msgfForm->loadImageFile(imgPath);
        n5Form->loadImageFile(imgPath);
    }

    mainWindow.show();
    return app.exec();
}
