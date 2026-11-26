#include "ui/ImageCanvas.h"

#include <QMouseEvent>
#include <QPainter>
#include <QResizeEvent>
#include <QWheelEvent>

#include <algorithm>

ImageCanvas::ImageCanvas(QWidget* parent)
    : QWidget(parent)
{
    setMouseTracking(true);
    setMinimumSize(140, 110);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void ImageCanvas::setImage(const QImage& img)
{
    const bool sizeChanged = (m_image.size() != img.size()) || img.isNull();
    m_image = img;
    if (sizeChanged) {
        m_autoFit = true;
    }
    if (m_autoFit) {
        updateFitScale();
    }
    update();
}

void ImageCanvas::setTitle(const QString& title)
{
    m_title = title;
    update();
}

void ImageCanvas::setPlaceholderText(const QString& text)
{
    m_placeholder = text;
    update();
}

void ImageCanvas::resetView()
{
    applyAutoFit();
}

void ImageCanvas::applyAutoFit()
{
    m_autoFit = true;
    updateFitScale();
    update();
}

void ImageCanvas::setZoomAndPan(double scale, const QPointF& offset)
{
    m_scale = std::clamp(scale, 0.05, 30.0);
    m_offset = offset;
    m_autoFit = false;
    update();
}

void ImageCanvas::updateFitScale()
{
    if (m_image.isNull() || width() <= 0 || height() <= 0) {
        return;
    }

    const double sx = static_cast<double>(std::max(1, width() - 16)) / m_image.width();
    const double sy = static_cast<double>(std::max(1, height() - 36)) / m_image.height();
    m_scale = std::min(sx, sy);
    if (m_scale <= 0.001) {
        m_scale = 1.0;
    }

    const double drawW = m_image.width() * m_scale;
    const double drawH = m_image.height() * m_scale;
    m_offset = QPointF((width() - drawW) / 2.0, (height() - drawH + 18.0) / 2.0);
}

void ImageCanvas::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    if (m_autoFit) {
        updateFitScale();
    }
}

void ImageCanvas::paintEvent(QPaintEvent* /*event*/)
{
    QPainter p(this);
    p.fillRect(rect(), QColor(28, 28, 30));

    if (!m_image.isNull()) {
        p.setRenderHint(QPainter::SmoothPixmapTransform, m_scale < 2.0);
        const QRectF targetRect(
            m_offset.x(),
            m_offset.y(),
            m_image.width() * m_scale,
            m_image.height() * m_scale
        );
        p.drawImage(targetRect, m_image);

        p.setPen(QColor(68, 68, 72));
        p.drawRect(targetRect);
    } else {
        p.setPen(QColor(135, 135, 142));
        p.drawText(rect(), Qt::AlignCenter, m_placeholder);
    }

    if (!m_title.isEmpty()) {
        QFont titleFont = font();
        titleFont.setPointSize(10);
        titleFont.setWeight(QFont::DemiBold);
        p.setFont(titleFont);
        const QFontMetrics fm = p.fontMetrics();
        const int textW = fm.horizontalAdvance(m_title);
        const int textH = fm.height();
        const QRect badgeRect(8, 8, textW + 14, textH + 6);

        p.setBrush(QColor(18, 18, 22, 215));
        p.setPen(QColor(65, 68, 76));
        p.drawRoundedRect(badgeRect, 4, 4);

        p.setPen(QColor(232, 234, 240));
        p.drawText(badgeRect, Qt::AlignCenter, m_title);
    }

    if (!m_image.isNull()) {
        const QString zoomStr = QString::number(static_cast<int>(m_scale * 100.0)) + "%";
        QFont zoomFont = font();
        zoomFont.setPointSize(9);
        p.setFont(zoomFont);
        const QFontMetrics fm = p.fontMetrics();
        const int textW = fm.horizontalAdvance(zoomStr);
        const QRect badgeRect(width() - textW - 16, height() - 23, textW + 10, 16);

        p.setBrush(QColor(18, 18, 22, 185));
        p.setPen(Qt::NoPen);
        p.drawRoundedRect(badgeRect, 3, 3);

        p.setPen(QColor(175, 180, 190));
        p.drawText(badgeRect, Qt::AlignCenter, zoomStr);
    }
}

void ImageCanvas::wheelEvent(QWheelEvent* event)
{
    if (m_image.isNull()) {
        return;
    }

    m_autoFit = false;
    const double factor = (event->angleDelta().y() > 0) ? 1.15 : (1.0 / 1.15);
    const QPointF mousePos = event->position();
    const QPointF imgPoint = (mousePos - m_offset) / m_scale;

    m_scale = std::clamp(m_scale * factor, 0.05, 30.0);
    m_offset = mousePos - imgPoint * m_scale;

    update();
    emit viewChanged(m_scale, m_offset);
}

void ImageCanvas::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        m_isDragging = true;
        m_lastMousePos = event->pos();
        setCursor(Qt::ClosedHandCursor);
    }
}

void ImageCanvas::mouseMoveEvent(QMouseEvent* event)
{
    if (m_isDragging) {
        m_autoFit = false;
        const QPoint delta = event->pos() - m_lastMousePos;
        m_lastMousePos = event->pos();
        m_offset += QPointF(delta.x(), delta.y());
        update();
        emit viewChanged(m_scale, m_offset);
    }
}

void ImageCanvas::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        m_isDragging = false;
        setCursor(Qt::ArrowCursor);
    }
}

void ImageCanvas::mouseDoubleClickEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton && !m_image.isNull()) {
        resetView();
        emit viewReset();
    }
}
