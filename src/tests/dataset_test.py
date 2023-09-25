from ..data_utils import *
import ipdb
import numpy as np
from ..visualization import animate_pose_matplotlib

def test_h36m_data_loading():

    PERSON_IDS = None
    ACTION_STR = None
    SUB_ACTION_ID = None
    RETURN_TENSOR = False


    data, meta_dir = load_data_h36m(return_tensor=True)

    # Compute some diagnostic information about the data
    num_sequences = len(data)
    data_shape= data[0][0].shape
    mean_frames_per_sequence = np.mean([len(seq) for seq in data])
    std_frames_per_sequence = np.std([len(seq) for seq in data])
    total_num_frames = 0
    mean_per_dim = np.zeros(99)
    std_per_dim = np.zeros(99)
    sum_per_dim = np.zeros(99)

    for seq in data:
        for frame in seq:
            total_num_frames += 1
            sum_per_dim += frame.numpy()

    mean_per_dim = sum_per_dim / total_num_frames
    active_angles = np.where(mean_per_dim != 0.0)
    zero_angles = np.where(mean_per_dim == 0.0)

    print(f'Test Data Loading:\n Num. Sequences: {num_sequences}\n Total Frames: {total_num_frames}\n Data Shape: {data_shape}\n Frames Per Sequence: mean: {mean_frames_per_sequence}, std: {std_frames_per_sequence}\n')
    print("\n".join([f' {i}: mean: {mean}' for i, mean in enumerate(mean_per_dim)]))
    print(f' Active Angles: {active_angles}\n Zero Angles: {zero_angles}')
    ipdb.set_trace()
    print('Test Finished!')

def test_ais_data_loading():

    dataset = AISDataset(
        seed_length=10,
        target_length=10,
        sequence_spacing=5,
        absolute_position=False,
        smooth=True
    )
    import ipdb; ipdb.set_trace()
    for i, data in enumerate(dataset):
        import ipdb; ipdb.set_trace()
        animate_pose_matplotlib(
                positions = (data.numpy(), data.numpy()),
                colors = ('g', 'g'),
                titles = ("test_1", "test_2"),
                fig_title = "Visualization Test",
                parents = SH_SKELETON_PARENTS,
                change_color_after_frame=(None, None),
                color_after_change='r',
                show_axis=True,
                overlay=True,
                fps=25,
                
            )
    
def test_data_augmentation():

    ##== Augmentation Parameters ==##
    params = {
        "normalize": False,
        "reverse_prob": 0.0,
        "snp_noise_prob": 0.0,
        "snp_noise_portion": [
            0.05,
            0.4
        ],
        "joint_cutout_prob": 0.0,
        "joint_cutout_portion": [
            1,
            4
        ],
        "timestep_cutout_prob": 0.0,
        "timestep_cutout_portion": [
            1,
            4
        ],
        "gaussian_noise_prob": 1.0,
        "gaussian_noise_std": 0.005
    }

    dataset = H36MDataset(
        seed_length=10,
        target_length=10,
        down_sampling_factor=2,
        stacked_hourglass=True,
        rot_representation="pos",
        sequence_spacing=0,
        return_label=False,
        is_train=True,
        debug=True
    )

    data_augmentor = get_data_augmentor(params)
    for i, seq in enumerate(dataset):
        seq = data_augmentor(seq.unsqueeze(0)).squeeze()
        animate_pose_matplotlib(
                positions = (seq.numpy(), seq.numpy()),
                colors = ('g', 'g'),
                titles = ("test_1", "test_2"),
                fig_title = "Visualization Test",
                parents = SH_SKELETON_PARENTS,
                change_color_after_frame=(None, None),
                color_after_change='r',
                overlay=True,
                show_axis=True,
                fps=25,
                
            )