from typing import Union

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import * 
from PyQt6.QtCore import pyqtProperty

from .toggle_switch import AnimatedToggle

"""
    Custom widgets that are used in the GUI.
"""


class ConfigEdit(QLineEdit):
    def __init__(self):
        super(ConfigEdit, self).__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignRight)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() == Qt.Key.Key_Return:
            self.clearFocus()
            self.setAlignment(Qt.AlignmentFlag.AlignRight)

class ConfigEntryEdit(QWidget):
    def __init__(self, name: str, default: Union[str, float, int]):
        super(ConfigEntryEdit, self).__init__()

        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        self.layout().setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.setObjectName(name)

        self.valueUpdated = pyqtSignal()

        self.layout().addWidget(QLabel(name), stretch=3)
        self.layout().addWidget(ConfigEdit(), stretch=1)

        self.line_edit = QLineEdit()
        self.line_edit.returnPressed.connect(self.fsignal)
        self.line_edit.setText(str(default))
        self.layout().addWidget(self.line_edit, stretch=1)

    def getValue(self):
        return self.line_edit.text()

    @pyqtSlot()
    def fsignal(self):
        self.valueUpdated.emit()

class ConfigEntrySwitch(QWidget):
    def __init__(self, name:str, default: bool):
        super(ConfigEntrySwitch, self).__init__()

        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        self.layout().setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.setObjectName(name)

        self.valueUpdated = pyqtSignal()

        self.layout().addWidget(QLabel(name), stretch=3)
        self.layout().addWidget(ConfigEdit(), stretch=1)

        self.switch = AnimatedToggle()
        self.switch.stateChanged.connect(self.fsignal)
        self.switch.setChecked(default)
        self.layout().addWidget(self.switch, stretch=1)

    def getValue(self):
        return self.switch.checkState()

    @pyqtSlot()
    def fsignal(self):
        self.valueUpdated.emit()


class ConfigEntrySlider(QWidget):
    def __init__(self, name: str, default: Union[float, int], min: Union[float, int], max: Union[float, int], step: Union[float, int]):
        super(ConfigEntrySlider, self).__init__()

        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        self.layout().setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.setObjectName(name)

        self.valueUpdated = pyqtSignal()

        self.layout().addWidget(QLabel(name), stretch=2)
        self.layout().addWidget(ConfigEdit(), stretch=3)

        self.slider = QSlider()
        self.slider.valueChanged.connect(self.fsignal)
        self.layout().addWidget(self.switch, stretch=1)

        self.slider_label = QLabel()
        self.layout().addWidget(self.slider_label, stretch=1)

    def getValue(self):
        return self.switch.checkState()

    @pyqtSlot()
    def fsignal(self):
        self.slider_label.setText(str(self.slider.value()))
        self.valueUpdated.emit()