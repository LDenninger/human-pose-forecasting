from typing import Union

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6 import uic
import sys
import json
import os
from pathlib import Path as P

from blackboard import Data, Config, Player
from resources import AnimatedToggle, ConfigEdit, ConfigEntrySwitch, ConfigEntrySlider, ConfigEntryEdit
from GLVisualizer import PoseVisualizer



class ConfigWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.build_widget()


    def build_widget(self):
        main_layout = QVBoxLayout()
        self.visualization_settings = QGroupBox("Visualization Settings")
        settings_layout = QVBoxLayout()

        # Set visualization settings ui
        self.interval_config = ConfigEntryEdit("interval", default=self.config.interval)
        self.interval_config.valueUpdated.connect(lambda: self.config.change_config({'interval': self.interval_config.getValue()}))
        settings_layout.addWidget(self.interval_config)

        self.show_joints_config = ConfigEntrySwitch("show_joints", default=self.config.show_joints)
        self.show_joints_config.valueUpdated.connect(lambda: self.config.change_config({'show_joints': self.show_joints_config.getValue()}))
        settings_layout.addWidget(self.show_joints_config)

        self.joint_dot_radius_config = ConfigEntryEdit("joint_dot_radius", default=self.config.joint_dot_radius)
        self.joint_dot_radius_config.valueUpdated.connect(lambda: self.config.change_config({'joint_dot_radius': self.joint_dot_radius_config.getValue()}))
        settings_layout.addWidget(self.joint_dot_radius_config)

        self.show_bones_config = ConfigEntrySwitch("show_bones", default=self.config.show_bones)
        self.show_bones_config.valueUpdated.connect(lambda: self.config.change_config({'show_bones': self.show_bones_config.getValue()}))
        settings_layout.addWidget(self.show_bones_config)

        self.bone_width_config = ConfigEntryEdit("bone_width", default=self.config.bone_width)
        self.bone_width_config.valueUpdated.connect(lambda: self.config.change_config({'bone_width': self.bone_width_config.getValue()}))
        settings_layout.addWidget(self.bone_width_config)

        self.center_at_hip_config = ConfigEntrySwitch("center_at_hip", default=self.config.center_at_hip)
        self.center_at_hip_config.valueUpdated.connect(lambda: self.config.change_config({'center_at_hip': self.center_at_hip_config.getValue()}))
        settings_layout.addWidget(self.center_at_hip_config)

        self.paint_coordinate_frame_config = ConfigEntrySwitch("paint_coordinate_frame", default=self.config.show_frames)
        self.paint_coordinate_frame_config.valueUpdated.connect(lambda: self.config.change_config({'paint_coordinate_frame': self.show_frames_config.getValue()}))
        settings_layout.addWidget(self.show_frames_config)

        self.x_rotation_view = ConfigEntrySlider("x_rotation_view", default=self.config.default_rotation[0])
        self.x_rotation_view.valueUpdated.connect(lambda: self)

        self.visualization_settings.setLayout(settings_layout)
        main_layout.addWidget(self.visualization_settings)


        # Set data settings ui
        self.data_settings = QGroupBox("Data Settings (disabled)")
        data_settings_layout = QGridLayout()
        self.data_settings.setLayout(data_settings_layout)
        self.data_fps_label, self.data_fps_edit = self.get_edit("FPS", default=self.config.data_fps)
        self.data_fps_edit.setReadOnly(True)
        data_settings_layout.addWidget(self.data_fps_label, 0, 0)
        data_settings_layout.addWidget(self.data_fps_edit, 0, 1)

        self.data_person_label, self.data_person_edit = self.get_edit("Person", default=self.config.data_person)
        self.data_person_edit.setReadOnly(True)
        data_settings_layout.addWidget(self.data_person_label, 1, 0)
        data_settings_layout.addWidget(self.data_person_edit, 1, 1)

        self.data_action_label, self.data_action_edit = self.get_edit("Action", default=self.config.data_action)
        self.data_action_edit.setReadOnly(True)
        data_settings_layout.addWidget(self.data_action_label, 2, 0)
        data_settings_layout.addWidget(self.data_action_edit, 2, 1)

        self.data_subaction_label, self.data_subaction_edit = self.get_edit("Subaction", default=self.config.data_subaction)
        self.data_subaction_edit.setReadOnly(True)
        data_settings_layout.addWidget(self.data_subaction_label, 3, 0)
        data_settings_layout.addWidget(self.data_subaction_edit, 3, 1)

        self.data_reset_button = QPushButton("Reset")
        self.data_reset_button.clicked.connect(self.data_config_reset)

        data_settings_layout.addWidget(self.data_reset_button, 4, 0)

        self.data_apply_button = QPushButton("Apply")
        self.data_apply_button.clicked.connect(lambda: self.config.change_config({'data_fps': self.data_fps_edit.text(),
                                                                                    'data_person': self.data_person_edit.text(),
                                                                                     'data_action': self.data_action_edit.text(),
                                                                                      'data_subaction': self.data_subaction_edit.text()}))
        data_settings_layout.addWidget(self.data_apply_button, 4, 1)
        self.disable_data_settings()
        main_layout.addWidget(self.data_settings)

        self.setLayout(main_layout)

    def get_slider(self, name: str, default: Union[int, float], low: Union[int, float], high: Union[int, float], step: Union[int, float]):
        label = QLabel(name)
        slider = QSlider()
        slider.setMinimum(low)
        slider.setMaximum(high)
        slider.setSingleStep(step)
        return label, slider

    def get_switch(self, name: str, default: bool):
        label = QLabel(name)
        switch = AnimatedToggle()
        switch.setChecked(default)
        return label, switch
    
    def get_edit(self, name: str, default: Union[int, float, str]):
        label = QLabel(name)
        edit = ConfigEdit()
        edit.setText(str(default))
        return label, edit
    

    @pyqtSlot()
    def data_config_reset(self):
        self.data_fps_edit.setText(str(self.config.data_fps))
        self.data_person_edit.setText(str(self.config.data_person))
        self.data_action_edit.setText(str(self.config.data_action))
        self.data_subaction_edit.setText(str(self.config.data_subaction))

    @pyqtSlot()
    def visualization_setting_reset(self):
        self.show_bones_switch.setChecked(self.config.show_bones)
        self.show_joints_switch.setChecked(self.config.show_joints)
        self.show_frames_switch.setChecked(self.config.show_frames)
        self.center_at_hip_switch.setChecked(self.config.center_at_hip)
        self.joint_dot_radius_slider.setValue(self.config.joint_dot_radius)
        self.bone_width_slider.setValue(self.config.bone_width)

    @pyqtSlot()
    def disable_data_settings(self):
        self.data_fps_edit.setReadOnly(True)
        self.data_person_edit.setReadOnly(True)
        self.data_action_edit.setReadOnly(True)
        self.data_subaction_edit.setReadOnly(True)
        self.data_reset_button.setEnabled(False)
        self.data_apply_button.setEnabled(False)

        opacity_effect = QGraphicsOpacityEffect(self)
        opacity_effect.setOpacity(0.5)
        self.data_reset_button.setGraphicsEffect(None)
        self.data_apply_button.setGraphicsEffect(None)
        self.data_reset_button.setGraphicsEffect(opacity_effect)
        self.data_apply_button.setGraphicsEffect(opacity_effect)

    @pyqtSlot()
    def enable_data_settings(self):
        self.data_fps_edit.setReadOnly(False)
        self.data_person_edit.setReadOnly(False)
        self.data_action_edit.setReadOnly(False)
        self.data_subaction_edit.setReadOnly(False)

        self.data_reset_button.setEnabled(True)
        self.data_apply_button.setEnabled(True)

        self.data_reset_button.setGraphicsEffect(None)
        self.data_apply_button.setGraphicsEffect(None)
class DataWidget(QWidget):
    def __init__(self):
        super().__init__()
        return


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.data = Data()
        self.player = Player()

        self.setWindowTitle("Pose Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()

        sshFile= P('resources') / "Combinear.qss"
        with open(sshFile,"r") as fh:
            self.setStyleSheet(fh.read())
        
    def init_ui(self):

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        self.config_widget = ConfigWidget()
        layout.addWidget(self.config_widget, stretch=1)

        self.pose_visualizer = PoseVisualizer()
        layout.addWidget(self.pose_visualizer, stretch=3)

        self.data_widget = DataWidget()
        layout.addWidget(self.data_widget, stretch=1)


    def init_menubar(self):
        pass



def main():
    pose_viewer = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    pose_viewer.exec()

def visualize_skeletons(skeleton_structure, data):
    pose_viewer = QApplication()
    window = MainWindow()
    window.show()

    

if __name__ == '__main__':
    main()