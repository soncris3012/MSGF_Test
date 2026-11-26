#pragma once

#include <QImage>
#include <QPoint>
#include <QPointF>
#include <QString>
#include <QWidget>

/**
 * @brief Lightweight QWidget image viewer with cursor-centered zoom,
 *        left-drag pan, double-click fit reset, and multi-canvas sync.
 */
class ImageCanvas : public QWidget {
    Q_OBJECT

public:
    explicit ImageCanvas(QWidget* parent = nullptr);

    void setImage(const QImage& img);
    void setTitle(const QString& title);
    void setPlaceholderText(const QString& text);
    void resetView();
    void applyAutoFit();

    void setZoomAndPan(double scale, const QPointF& offset);
    [[nodiscard]] bool hasImage() const noexcept { return !m_image.isNull(); }

signals:
    void viewChanged(double scale, const QPointF& offset);
    void viewReset();

protected:
    void paintEvent(QPaintEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    void updateFitScale();

    QImage m_image;
    QString m_title;
    QString m_placeholder = QStringLiteral("No image loaded");
    double m_scale = 1.0;
    QPointF m_offset = QPointF(0.0, 0.0);
    bool m_isDragging = false;
    QPoint m_lastMousePos;
    bool m_autoFit = true;
};
