# General Information
DATASET_PATH = 'data/h3.6m/dataset'
DATASET_PERSONS = [1,5,6,7,8,9,11]
DATASET_ACTIONS = ['directions', 'discussion', 'eating', 'greeting', 'posing', 'phoning', 'purchases', 'sitting', 'sittingdown', 'smoking', 'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether']

# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = [1,5,6,7,8]
TEST_SUBJECTS  = [9,11]

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee' # only x
H36M_NAMES[3]  = 'RAnkle'
H36M_NAMES[4]  = 'RToe' # only x
H36M_NAMES[5]  = 'Site' # dead
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee' # only x
H36M_NAMES[8]  = 'LAnkle'
H36M_NAMES[9]  = 'LToe' # only x
H36M_NAMES[10] = 'Site' # dead
H36M_NAMES[11] = 'Spine' # parent: hip
H36M_NAMES[12] = 'Spine1' # parent: spine
H36M_NAMES[13] = 'Thorax' # parent: spine1
H36M_NAMES[14] = 'Neck' # parent: thorax
H36M_NAMES[15] = 'Head' # no angle, final joint
H36M_NAMES[16] = 'LShoulder' # used as linkage of LShoulder, parent: spine1
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow' # only x
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[20] = 'LThumb' # no angle, final joint
H36M_NAMES[21] = 'Site' # dead
H36M_NAMES[22] = 'L_Wrist_End' # no angle, final joint
H36M_NAMES[24] = 'RShoulder' # used as linkage of RShoulder, parent: spine1
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'
H36M_NAMES[28] = 'RThumb' # no angle, final joint
H36M_NAMES[29] = 'Site' # dead
H36M_NAMES[30] = 'R_Wrist_End' # no angle, final joint
H36M_NAMES[31] = 'Site' # dead

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
    'angle_ind': [[5, 6, 4],
            [8, 9, 7],
            [11, 12, 10],
            [14, 15, 13],
            [17, 18, 16],
            [],
            [20, 21, 19],
            [23, 24, 22],
            [26, 27, 25],
            [29, 30, 28],
            [], #10
            [32, 33, 31],
            [35, 36, 34],
            [38, 39, 37],
            [41, 42, 40],
            [],#15
            [44, 45, 43],
            [47, 48, 46],
            [50, 51, 49],
            [53, 54, 52],
            [56, 57, 55], #20
            [],
            [59, 60, 58],
            [],
            [62, 63, 61],
            [65, 66, 64],
            [68, 69, 67],
            [71, 72, 70],
            [74, 75, 73],
            [],
            [77, 78, 76],
            []],
    'moving_joints': [0, 1, 2, 3, 4 ]
}


