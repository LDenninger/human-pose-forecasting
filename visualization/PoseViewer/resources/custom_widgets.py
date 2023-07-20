from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import * 
from PyQt6.QtCore import pyqtProperty

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
        