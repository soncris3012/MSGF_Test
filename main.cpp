#include "ui/MsgfForm.h"

#include <QApplication>
#include <QMainWindow>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("MSGF-TIR"));
    app.setOrganizationName(QStringLiteral("MSGF_Test"));

    QMainWindow mainWindow;
    mainWindow.setWindowTitle(
        QStringLiteral("MSGF-TIR: Multi-Scale Guided Filtering Thermal Enhancement (Fire 2024, 7, 192)")
    );
    auto* form = new MsgfForm(&mainWindow);
    mainWindow.setCentralWidget(form);
    mainWindow.resize(1480, 880);

    // Optional CLI argument to preload an image on launch
    if (argc > 1) {
        form->loadImageFile(QString::fromLocal8Bit(argv[1]));
    }

    mainWindow.show();
    return app.exec();
}
