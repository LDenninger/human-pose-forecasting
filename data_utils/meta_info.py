import numpy as np

# General Information
DATASET_PATH = 'data/h3.6m/dataset'
DATASET_PERSONS = [1,5,6,7,8,9,11]
DATASET_ACTIONS = ['directions', 'discussion', 'eating', 'greeting', 'posing', 'phoning', 'purchases', 'sitting', 'sittingdown', 'smoking', 'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether']

# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = [1,5,6,7,8]
TEST_SUBJECTS  = [9,11]

DEBUG_SPLIT = [1,]

# Joints in H3.6M -- data has 32 joints, but only 17 that move;
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'hip'
H36M_NAMES[1]  = 'rHip'
H36M_NAMES[2]  = 'rKnee' # only x
H36M_NAMES[3]  = 'rAnkle'
H36M_NAMES[4]  = 'rToe' # only x
H36M_NAMES[5]  = 'site' # dead
H36M_NAMES[6]  = 'lHip'
H36M_NAMES[7]  = 'lKnee' # only x
H36M_NAMES[8]  = 'lAnkle'
H36M_NAMES[9]  = 'lToe' # only x
H36M_NAMES[10] = 'site' # dead
H36M_NAMES[11] = 'spine' # parent: hip
H36M_NAMES[12] = 'spine1' # parent: spine
H36M_NAMES[13] = 'thorax' # parent: spine1
H36M_NAMES[14] = 'neck' # parent: thorax
H36M_NAMES[15] = 'head' # no angle, final joint
H36M_NAMES[16] = 'lShoulderAnchor' # used as linkage of lShoulder, parent: spine1
H36M_NAMES[17] = 'lShoulder'
H36M_NAMES[18] = 'lElbow' # only x
H36M_NAMES[19] = 'lWrist'
H36M_NAMES[20] = 'lThumb' # no angle, final joint
H36M_NAMES[21] = 'site' # dead
H36M_NAMES[22] = 'lWristEnd' # no angle, final joint
H36M_NAMES[24] = 'rShoulderAnchor' # used as linkage of rShoulder, parent: spine1
H36M_NAMES[25] = 'rShoulder'
H36M_NAMES[26] = 'rElbow' # only x
H36M_NAMES[27] = 'rWrist'
H36M_NAMES[28] = 'rThumb' # no angle, final joint
H36M_NAMES[29] = 'site' # dead
H36M_NAMES[30] = 'rWristEnd' # no angle, final joint
H36M_NAMES[31] = 'site' # dead

H36M_NAMES_TO_IND = {
    'hip': 0,
    'rHip': 1,
    'rKnee': 2,
    'rFoot': 3,
    'rFootTip': 4,
    'lHip': 6,
    'lKnee': 7,
    'lFoot': 8,
    'lFootTip': 9,
    'neck': 12,
    'head': 15,
    'lShoulder': 17,
    'lElbow': 18,
    'lWrist': 19,
    'rElbow': 26,
    'rShoulder': 25,
    'rWrist': 27,
    'thorax': 13,
    'spine': 12,
}
H36M_SKELETON_STRUCTURE = {
    0: ('hip', 'root'),
    1: ('rHip', 'hip'),
    2: ('rKnee', 'rHip'),
    3: ('rAnkle', 'rKnee'),
    4: ('rToe', 'rAnkle'),
    5: ('lHip', 'hip'),
    6: ('lKnee', 'lHip'),
    7: ('lAnkle', 'lKnee'),
    8: ('lToe', 'lAnkle'),
    9: ('spine', 'hip'),
    10: ('spine1', 'spine'),
    11: ('thorax', 'spine1'),
    12: ('neck', 'thorax'),
    13: ('head', 'neck'),
    14: ('lShoulderAnchor', 'spine1'),
    15: ('lShoulder', 'lShoulderAnchor'),
    16: ('lElbow', 'lShoulder'),
    17: ('lWrist', 'lElbow'),
    18: ('lThumb', 'lWrist'),
    20: ('lWristEnd', 'lWrist'),
    21: ('rShoulderAnchor', 'spine1'),
    22: ('rShoulder', 'rShoulderAnchor'),
    23: ('rElbow', 'rShoulder'),
    24: ('rWrist', 'rElbow'),
    25: ('rThumb', 'rWrist'),
    26: ('rWristEnd', 'rWrist')
}

SKELETON_H36M_BONE_LENGTH = [0.000000,0.000000,0.000000,-132.948591,0.000000,0.000000,0.000000,-442.894612,0.000000,0.000000,-454.206447,0.000000,0.000000,0.000000,162.767078,0.000000,0.000000,74.999437,132.948826,0.000000,0.000000,0.000000,-442.894413,0.000000,0.000000,-454.206590,0.000000,0.000000,0.000000,162.767426,0.000000,0.000000,74.999948,0.000000,0.100000,0.000000,0.000000,233.383263,0.000000,0.000000,257.077681,0.000000,0.000000,121.134938,0.000000,0.000000,115.002227,0.000000,0.000000,257.077681,0.000000,0.000000,151.034226,0.000000,0.000000,278.882773,0.000000,0.000000,251.733451,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999627,0.000000,100.000188,0.000000,0.000000,0.000000,0.000000,0.000000,257.077681,0.000000,0.000000,151.031437,0.000000,0.000000,278.892924,0.000000,0.000000,251.728680,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999888,0.000000,137.499922,0.000000,0.000000,0.000000,0.000000]
SKELETON_H36M_PARENT_IDS = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11 ,12,13,14,12,16,17,18,19,20,19,22,12,24,25,26,27,28,27,30]
BASELINE_FKL_IND = np.split(np.arange(4, 100) - 1, 32)


# Stacked Hourglass produces 16 joints. These are the names.
SH_NAMES = ['']*16
SH_NAMES[0]  = 'RFoot'
SH_NAMES[1]  = 'RKnee'
SH_NAMES[2]  = 'RHip'
SH_NAMES[3]  = 'LHip'
SH_NAMES[4]  = 'LKnee'
SH_NAMES[5]  = 'LFoot'
SH_NAMES[6]  = 'Hip'
SH_NAMES[7]  = 'Spine'
SH_NAMES[8]  = 'Thorax'
SH_NAMES[9]  = 'Head'
SH_NAMES[10] = 'RWrist'
SH_NAMES[11] = 'RElbow'
SH_NAMES[12] = 'RShoulder'
SH_NAMES[13] = 'LShoulder'
SH_NAMES[14] = 'LElbow'
SH_NAMES[15] = 'LWrist'



# Skeleton Models

SKELETON_H36M_MODEL = {
    'joint_names': H36M_NAMES,
    'name_to_ind': H36M_NAMES_TO_IND,
    'parent_ids': [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11 ,12,13,14,12,16,17,18,19,20,19,22,12,24,25,26,27,28,27,30],
    'bone_length': [0.000000,0.000000,0.000000,-132.948591,0.000000,0.000000,0.000000,-442.894612,0.000000,0.000000,-454.206447,0.000000,0.000000,0.000000,162.767078,0.000000,0.000000,74.999437,132.948826,0.000000,0.000000,0.000000,-442.894413,0.000000,0.000000,-454.206590,0.000000,0.000000,0.000000,162.767426,0.000000,0.000000,74.999948,0.000000,0.100000,0.000000,0.000000,233.383263,0.000000,0.000000,257.077681,0.000000,0.000000,121.134938,0.000000,0.000000,115.002227,0.000000,0.000000,257.077681,0.000000,0.000000,151.034226,0.000000,0.000000,278.882773,0.000000,0.000000,251.733451,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999627,0.000000,100.000188,0.000000,0.000000,0.000000,0.000000,0.000000,257.077681,0.000000,0.000000,151.031437,0.000000,0.000000,278.892924,0.000000,0.000000,251.728680,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999888,0.000000,137.499922,0.000000,0.000000,0.000000,0.000000],
    'angle_ind': [[5, 6, 4], #hip
            [8, 9, 7], #rhip
            [11, 12, 10],
            [14, 15, 13],
            [17, 18, 16],
            [],# 5
            [20, 21, 19], #lhip
            [23, 24, 22],
            [26, 27, 25],
            [29, 30, 28],
            [], #10
            [32, 33, 31], # spine
            [35, 36, 34],
            [38, 39, 37],
            [41, 42, 40],
            [],#15
            [44, 45, 43], # lShoulderAnchor
            [47, 48, 46],
            [50, 51, 49],
            [53, 54, 52],
            [56, 57, 55], #20
            [],
            [59, 60, 58], # lWristEnd
            [],
            [62, 63, 61], # rShoulderAnchor
            [65, 66, 64], #25
            [68, 69, 67],
            [71, 72, 70],
            [74, 75, 73],
            [],
            [77, 78, 76], #30
            []],
    'moving_joints': [0, 1, 2, 3, 4 ]
}


rotInd = [[6, 5, 4], #0 hip
              [9, 8, 7], # rhip
              [12, 11, 10], # rknee
              [15, 14, 13], # rankle
              [18, 17, 16], #rtoe
              [21, 20, 19], #5
              [],
              [24, 23, 22],
              [27, 26, 25],
              [30, 29, 28],
              [33, 32, 31], #10
              [36, 35, 34],
              [],
              [39, 38, 37],
              [42, 41, 40],
              [45, 44, 43], #15
              [48, 47, 46],
              [51, 50, 49],
              [54, 53, 52],
              [],
              [57, 56, 55], #20
              [60, 59, 58],
              [63, 62, 61],
              [66, 65, 64],
              [69, 68, 67],
              [72, 71, 70], #25
              [],
              [75, 74, 73],
              [],
              [78, 77, 76],
              [81, 80, 79], #30
              [84, 83, 82],
              [87, 86, 85],
              [90, 89, 88],
              [93, 92, 91],
              [],
              [96, 95, 94],
              []]