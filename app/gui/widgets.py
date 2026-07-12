"""
widgets.py
----------
Reusable GUI widgets and helper factories used by the main window.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette, QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QWidget,
)


# ------------------------------------------------------------------ #
# Stat card (label + value pair)                                     #
# ------------------------------------------------------------------ #

class StatCard(QFrame):
    """A small rounded card displaying a *label* and a numeric *value*.

    Args:
        title: Static description text shown above the value.
        parent: Optional parent widget.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        from PySide6.QtWidgets import QVBoxLayout

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        self._title_label = QLabel(title, self)
        self._title_label.setObjectName("StatCardTitle")
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._value_label = QLabel("0", self)
        self._value_label.setObjectName("StatCardValue")
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self._title_label)
        layout.addWidget(self._value_label)

    def set_value(self, value: int) -> None:
        """Update the displayed numeric *value*."""
        self._value_label.setText(str(value))


# ------------------------------------------------------------------ #
# Scrollable log widget                                              #
# ------------------------------------------------------------------ #

class LogWidget(QPlainTextEdit):
    """Read-only, auto-scrolling plain-text log widget.

    Args:
        parent: Optional parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("LogWidget")
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        monospace = QFont("Consolas", 9)
        monospace.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(monospace)

    def append_line(self, text: str) -> None:
        """Append *text* as a new line and scroll to the bottom."""
        self.appendPlainText(text)
        self.moveCursor(QTextCursor.MoveOperation.End)

    def clear_log(self) -> None:
        """Remove all text from the log."""
        self.clear()


# ------------------------------------------------------------------ #
# Styled push-button factory                                         #
# ------------------------------------------------------------------ #

def make_button(
    label: str,
    object_name: str,
    parent: QWidget | None = None,
    min_width: int = 110,
) -> QPushButton:
    """Create a :class:`QPushButton` with consistent styling hooks.

    Args:
        label:       Button text.
        object_name: Qt object name used for CSS targeting.
        parent:      Optional parent widget.
        min_width:   Minimum button width in pixels.

    Returns:
        Configured :class:`QPushButton`.
    """
    btn = QPushButton(label, parent)
    btn.setObjectName(object_name)
    btn.setMinimumWidth(min_width)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    return btn


# ------------------------------------------------------------------ #
# Styled progress bar factory                                        #
# ------------------------------------------------------------------ #

def make_progress_bar(parent: QWidget | None = None) -> QProgressBar:
    """Create a styled :class:`QProgressBar`.

    Args:
        parent: Optional parent widget.

    Returns:
        Configured :class:`QProgressBar`.
    """
    bar = QProgressBar(parent)
    bar.setObjectName("MainProgressBar")
    bar.setMinimum(0)
    bar.setMaximum(100)
    bar.setValue(0)
    bar.setTextVisible(True)
    bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
    bar.setFixedHeight(22)
    return bar
