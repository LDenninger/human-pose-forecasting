from PyQt6.QtWidgets import *
from PyQt6.QtCore import *

import os
import json

"""
    Blackboard:
        This file contains all modules that hold the central information about the application.
        The visualizer is structured such that we have a singleton config and data object that can be accessed centralized by different instances.
        In this way it works like a blackboard where everybody can post and access their information.
"""



class Config(QObject):

    """
        Config object that holds all parameters/settings regarding the pose viewer.
    """
    ##-- Signals --##
    configUpdated = pyqtSignal()

    ##-- Config Parameters --##

    # Visualization parameters
    interval = 0.1
    origin = [0., 0., 0.]
    axis_limits = [5.0, 5.0, 5.0]
    num_ticks: int = 10
    viewpoint: list = [0.0, 0.0]
    camera_distance: float = 0.0
    
    annotate_joints = False
    show_bones = True
    show_joints = True
    show_frames = False
    center_at_hip = True

    joint_dot_radius = 0.05
    bone_width = 0.02
    bone_color = [255,0,0]

    color_scheme: list = None
    paint_coordinate_frame: bool = True
    paint_grid: bool = False

    


    # Data parameters
    data_fps: int = None
    data_person: int = None
    data_action: str = None
    data_subaction: int = None

    # Ensure singleton object
    _self = None
    def __new__(cls):
        if cls._self is None:
            cls._self = super().__new__(cls)
        return cls._self

    def __init__(self):
        super().__init__()
        self.load_default_config()



    def change_config(self, data):
        for key, value in data.items():
            if hasattr(Data, key):
                setattr(Data, key, value)
        self.configUpdated.emit()

    def load_default_config(self):
        with open(os.path.join('resources', 'pw_default_config.json'), 'r') as f:
            default_config = json.load(f)
        for key, value in default_config.items():
            if hasattr(Data, key):
                setattr(Data, key, value)


    

class Data(QObject):

    ##-- Signals --##
    newData = pyqtSignal()
    newSkeleton = pyqtSignal()


    ##-- Data Parameters --##
    cur_person: int = None
    cur_action: str = None
    cur_subaction: int = None
    cur_data = None
    cur_start_frame: int = None
    cur_frame: int = None
    cur_end_frame: int = None
    cur_sequence_length: int = None

    skeleton_data = {}
    skeleton_datatypes = {}
    label_to_index = {}
    index_to_label = {}
    skeleton_structre = {}
    current_data = None


    # Ensure singleton object
    _self = None
    def __new__(cls):
        if cls._self is None:
            cls._self = super().__new__(cls)
        return cls._self

    def __init__(self):
        super().__init__()

    def get_current_timeframe(self):
        return Data.cur_data

    ##-- Visualization Data --##
    def setData(self, person: int = None, action: str = None, subaction: int = None, frame: int = 0):
        if person is not None:
            Data.cur_person = person
        if action is not None:
            Data.cur_action = action
        if subaction is not None:
            Data.cur_subaction = subaction
        Data.cur_frame = frame
        Data.cur_data = {None}
        self._update_current_data()
        Data.newData.emit()

    @pyqtSlot()
    def next_frame(self):
        Data.cur_frame += 1
        self._update_current_data()
        Data.newData.emit()

    @pyqtSlot()
    def previous_frame(self):
        Data.cur_frame -= 1
        self._update_current_data()
        Data.newData.emit()

    @pyqtSlot()
    def set_frame(self, frame):
        Data.cur_frame = frame
        self._update_current_data()
        Data.newData.emit()

    def _update_current_data(self):
        assert Data.cur_person is not None, "Person must be defined to load data"
        assert Data.cur_action is not None, "Action must be defined to load data"
        assert Data.cur_subaction is not None, "Subaction must be defined to load data"
        assert Data.cur_frame is not None, "Frame must be defined to load data"
        for sk_name, data in Data.skeleton_data.items():
            index = Data.label_to_index[sk_name][Data.cur_person ][Data.cur_action][Data.cur_subaction]
            Data.cur_data[sk_name] = Data.skeleton_data[sk_name][index][Data.cur_frame]

    ##-- Data Loading --##
    def register_skeleton(self, name, skeleton_structure):
        Data.skeleton_data[name] = []
        Data.skeleton_structre[name] = skeleton_structure
        Data.newSkeleton.emit()

    def receive_dataset(self,
                            skeleton_name: str,
                             data: list,
                              label_to_index: dict,
                               index_to_list: list = None,):
        assert skeleton_name in Data.skeleton_data.keys(), 'Please register the skeleton first'
        Data.skeleton_data[skeleton_name] = data
        self.label_to_index[skeleton_name] = label_to_index
        if index_to_list is not None:
            Data.index_to_label[skeleton_name] = index_to_list

    def receive_sequence(self, 
                        skeleton_name: str,
                         data: list,
                          label: dict):
        assert skeleton_name in Data.skeleton_data.keys(), 'Please register the skeleton first'
        Data.skeleton_data[skeleton_name].append(data)
        if label['person_id'] not in Data.label_to_index.keys():
            Data.label_to_index[skeleton_name] = {}
        if label['action_str'] not in Data.label_to_index[skeleton_name][label['person_id']].keys():
            Data.label_to_index[skeleton_name][label['person_id']][label['action_str']] = {}
        Data.label_to_index[skeleton_name][label['person_id']][label['action_str']][label['subaction_id']] = len(Data.skeleton_data[skeleton_name]) - 1


class Player(QTimer):
    _self = None
    def __new__(cls):
        if cls._self is None:
            cls._self = super().__new__(cls)
        return cls._self
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.data = Data()
        self.setInterval(Config.interval)
        self.timeout.connect(self.data.next_frame)

    @pyqtSlot()
    def set_frame(self, frame):
        self.stop()
        self.data.set_frame(frame)
        self.start()
    
