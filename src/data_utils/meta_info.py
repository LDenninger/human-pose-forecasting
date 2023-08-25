import numpy as np

"""
    This file holds all meta information about the dataset.
    Since there is very little low-level documentation, this file might give a deeper insight into the dataset.
    It also provides easier access to the datasets for other modules and functions.

"""

#####===== H36M General Information =====#####
DATASET_PATH = 'data/h3.6m/dataset'
DATASET_PERSONS = [1,5,6,7,8,9,11]
DATASET_ACTIONS = ['directions', 'discussion', 'eating', 'greeting', 'posing', 'phoning', 'purchases', 'sitting', 'sittingdown', 'smoking', 'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether']

#####===== Official Data Splits =====#####
TRAIN_SUBJECTS = [1,5,6,7,8]
TEST_SUBJECTS  = [9,11]
DEBUG_SPLIT = [1,]

#####===== H36M Skeleton Information =====#####
# Number Joints: 32
# Active Joints: 27
# Moving Joints: 17
###=== Joint Names ===###
# The site joints are not used at all and are ignored throughout the project
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
H36M_NAMES[23] = 'site' # dead
H36M_NAMES[24] = 'rShoulderAnchor' # used as linkage of rShoulder, parent: spine1
H36M_NAMES[25] = 'rShoulder'
H36M_NAMES[26] = 'rElbow' # only x
H36M_NAMES[27] = 'rWrist'
H36M_NAMES[28] = 'rThumb' # no angle, final joint
H36M_NAMES[29] = 'site' # dead
H36M_NAMES[30] = 'rWristEnd' # no angle, final joint
H36M_NAMES[31] = 'site' # dead

###=== Joint Indices ===###
# These indices are used to retrieve the joint data from the 99-dimensional vector
# Positions [0,1,2] are the position of the hip with respect to the root frame

H36M_ANGLE_INDICES = {
    0: [3, 4, 5],
    1: [6, 7, 8],
    2: [9, 10, 11],
    3: [12, 13, 14],
    4: [15, 16, 17],
    5: [18, 19, 20],
    6: [21, 22, 23],
    7: [24, 25, 26],
    8: [27, 28, 29],
    9: [30, 31, 32],
    10: [33, 34, 35],
    11: [36, 37, 38],
    12: [39, 40, 41],
    13: [42, 43, 44],
    14: [45, 46, 47],
    15: [48, 49, 50],
    16: [51, 52, 53],
    17: [54, 55, 56],
    18: [57, 58, 59],
    19: [60, 61, 62],
    20: [63, 64, 65],
    21: [66, 67, 68],
    22: [69, 70, 71],
    23: [72, 73, 74],
    24: [75, 76, 77],
    25: [78, 79, 80],
    26: [81, 82, 83],
    27: [84, 85, 86],
    28: [87, 88, 89],
    29: [90, 91, 92],
    30: [93, 94, 95],
    31: [96, 97, 98]
}
BASELINE_FKL_IND = np.split(np.arange(4, 100) - 1, 32)

###===== Skeleton Structure ===###
# This is the skeleton structure defined within the H36M dataset
# Format: {frame_id: (frame_name, parent_frame_name}, ...}

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

###=== Bone Length ===###
# Length of the limbs connecting the joints
H36M_BONE_LENGTH = [0.000000,0.000000,0.000000,-132.948591,0.000000,0.000000,0.000000,-442.894612,0.000000,0.000000,-454.206447,0.000000,0.000000,0.000000,162.767078,0.000000,0.000000,74.999437,132.948826,0.000000,0.000000,0.000000,-442.894413,0.000000,0.000000,-454.206590,0.000000,0.000000,0.000000,162.767426,0.000000,0.000000,74.999948,0.000000,0.100000,0.000000,0.000000,233.383263,0.000000,0.000000,257.077681,0.000000,0.000000,121.134938,0.000000,0.000000,115.002227,0.000000,0.000000,257.077681,0.000000,0.000000,151.034226,0.000000,0.000000,278.882773,0.000000,0.000000,251.733451,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999627,0.000000,100.000188,0.000000,0.000000,0.000000,0.000000,0.000000,257.077681,0.000000,0.000000,151.031437,0.000000,0.000000,278.892924,0.000000,0.000000,251.728680,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999888,0.000000,137.499922,0.000000,0.000000,0.000000,0.000000]



#####===== Additional Skeleton Definitions =====#####
###=== Stacked Hourglass Skeleton ===###
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
