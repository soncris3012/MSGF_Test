#pragma once

#include <QString>
#include <QTableWidgetItem>

/**
 * @brief QTableWidgetItem that displays formatted numeric text (or Mean +/- Std)
 *        while sorting strictly by the underlying double stored in Qt::UserRole.
 */
class NumericTableItem : public QTableWidgetItem {
public:
    explicit NumericTableItem(double sortValue,
                              int precision = 2,
                              const QString& suffix = QString())
        : QTableWidgetItem(QString::number(sortValue, 'f', precision) + suffix)
    {
        setData(Qt::UserRole, sortValue);
        setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
    }

    NumericTableItem(double sortValue, const QString& displayText)
        : QTableWidgetItem(displayText)
    {
        setData(Qt::UserRole, sortValue);
        setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
    }

    bool operator<(const QTableWidgetItem& other) const override
    {
        const QVariant a = data(Qt::UserRole);
        const QVariant b = other.data(Qt::UserRole);
        if (a.isValid() && b.isValid()) {
            return a.toDouble() < b.toDouble();
        }
        return QTableWidgetItem::operator<(other);
    }
};
