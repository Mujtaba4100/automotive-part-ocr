"""
widgets.py
----------
Reusable GUI widgets and helper factories used by the main window.
"""

from __future__ import annotations

import datetime
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QTextEdit,
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
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        self._title_label = QLabel(title, self)
        self._title_label.setObjectName("StatCardTitle")
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._value_label = QLabel("0", self)
        self._value_label.setObjectName("StatCardValue")
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self._title_label)
        layout.addWidget(self._value_label)

    def set_value(self, value: int | float | str) -> None:
        """Update the displayed value with formatted output."""
        if isinstance(value, float):
            self._value_label.setText(f"{value:.1f}")
        else:
            self._value_label.setText(str(value))


# ------------------------------------------------------------------ #
# Scrollable log widget                                              #
# ------------------------------------------------------------------ #

class LogWidget(QTextEdit):
    """Read-only, auto-scrolling log widget with rich HTML coloring and timestamps.

    Args:
        parent: Optional parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("LogWidget")
        self.setReadOnly(True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        monospace = QFont("Consolas", 10)
        monospace.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(monospace)

    def append_line(self, text: str) -> None:
        """Append *text* as formatted HTML with timestamp and syntax color."""
        now = datetime.datetime.now().strftime("%H:%M:%S")
        timestamp = f"<span style='color: #4b5563; font-weight: 500;'>[{now}]</span> "
        
        # Determine color based on content indicators
        color = "#cbd5e1"  # Default slate-300
        
        lower_text = text.lower()
        if "✓" in text or "copied" in lower_text or "finished" in lower_text:
            color = "#10b981"  # Green
        elif "error" in lower_text or "failed" in lower_text or "exception" in lower_text:
            color = "#ef4444"  # Red
        elif "✗" in text or "no oem" in lower_text or "cancelled" in lower_text:
            color = "#f43f5e"  # Light Red / Crimson
        elif "⚠" in text or "warning" in lower_text or "cancelling" in lower_text:
            color = "#f59e0b"  # Amber / Orange
        elif "scanning" in lower_text or "ready" in lower_text or "starting" in lower_text or "output folder" in lower_text:
            color = "#3b82f6"  # Blue

        # Split multiple lines to keep indentation clean
        lines = text.split("\n")
        formatted_lines = []
        for idx, line in enumerate(lines):
            escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if idx == 0:
                prefix = timestamp
            else:
                # Align consecutive lines with the timestamp block width
                prefix = "&nbsp;" * 12
            
            # Format numbers or paths to highlight them
            formatted_lines.append(f"{prefix}<span style='color: {color};'>{escaped}</span>")

        html_chunk = "<br>".join(formatted_lines)
        self.append(html_chunk)
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
    bar.setFixedHeight(24)
    return bar
