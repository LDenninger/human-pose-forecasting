import torch
from torch.utils.data import DataLoader
from ..data_utils import (
    SkeletonModel32,
    H36MDataset,
    H36M_SKELETON_STRUCTURE, H36M_NON_REDUNDANT_SKELETON_STRUCTURE,
    baseline_forward_kinematics,
    convert_baseline_representation,
    expmap2rotmat,
    axis_angle_to_matrix,
    h36m_forward_kinematics,
    get_annotated_dictionary,
    convert_s26_to_s21, convert_s26_to_s16, convert_s21_to_s26, convert_s21_to_s16,
    parse_h36m_to_s26
)

from ..visualization import visualize_skeleton, compare_skeleton
import time


def test_skeleton32_model():
    skeleton = SkeletonModel32()

    dataset = H36MDataset(
        seed_length=40,
        target_length=1,
        down_sampling_factor=2,
        sequence_spacing=0,
        return_label=True,
        is_train=True,
        debug=True
    )

    for seq, label in dataset:
        position_list = []
        baseline_position_list = []

        for i in range(seq.shape[0]):
            # import ipdb; ipdb.set_trace()
            skeleton(seq[i])
            joint_positions = skeleton.getJointPositions(incl_names=True)
            # import ipdb; ipdb.set_trace()
            baseline_positions, baseline_out = baseline_forward_kinematics(
                angles=seq[i]
            )
            baseline_positions = convert_baseline_representation(baseline_positions)

            position_list.append(joint_positions)
            baseline_position_list.append(baseline_positions)
        import ipdb

        ipdb.set_trace()
        title_info = f" Person: {label['person_id']}, Action: {label['action_str']}, SubAction: {label['sub_action_id']}"
        time.sleep(1)
        # visualize_skeleton(position_list, H36M_SKELETON_STRUCTURE, title_text = title_info)
        time.sleep(3)
        # visualize_skeleton(baseline_position_list, H36M_SKELETON_STRUCTURE, title_text = title_info)
        compare_skeleton(
            baseline_position_list,
            position_list,
            H36M_SKELETON_STRUCTURE,
            H36M_SKELETON_STRUCTURE,
            title_text=title_info,
        )

    print("Done!")

def test_h36m_forward_kinematics():
    dataset = H36MDataset(
        seed_length=40,
        target_length=1,
        down_sampling_factor=2,
        sequence_spacing=0,
        return_label=False,
        raw_data=True,
        is_train=True,
        debug=True
    )
    skeleton = SkeletonModel32()

    for seq in dataset:
        position_list = []
        baseline_position_list = []
        seq_positions, seq_rotation = h36m_forward_kinematics(seq, representation='axis', hip_as_root=False)

        for i in range(seq.shape[0]):
            # import ipdb; ipdb.set_trace()
            joint_positions = seq_positions[i]
            joint_positions = get_annotated_dictionary(joint_positions, 's26')
            # import ipdb; ipdb.set_trace()
            baseline_positions, baseline_out = baseline_forward_kinematics(
                angles=seq[i]
            )
            baseline_positions = convert_baseline_representation(baseline_positions)

            position_list.append(joint_positions)
            baseline_position_list.append(baseline_positions)
        import ipdb

        ipdb.set_trace()
        # visualize_skeleton(position_list, H36M_SKELETON_STRUCTURE, title_text = title_info)
        time.sleep(3)
        # visualize_skeleton(baseline_position_list, H36M_SKELETON_STRUCTURE, title_text = title_info)
        compare_skeleton(
            position_list,
            position_list,
            H36M_SKELETON_STRUCTURE,
            H36M_SKELETON_STRUCTURE,

        )

    print("Done!")


def test_s21_skeleton():
    # Load the dataset
    dataset = H36MDataset(
        seed_length=40,
        target_length=1,
        down_sampling_factor=2,
        sequence_spacing=0,
        return_label=False,
        is_train=True,
        debug=True
    )
    import ipdb; ipdb.set_trace()
    for seq in dataset:
        position_list = []
        baseline_position_list = []
        s26_representation = parse_h36m_to_s26(seq)
        s21_representation = convert_s26_to_s21(s26_representation, relative=True, rot_representation='axis')
        seq_positions, seq_rotation = s21_forward_kinematics(s21_representation, hip_as_root=True, representation='axis')

        for i in range(seq.shape[0]):
            # import ipdb; ipdb.set_trace()
            joint_positions = seq_positions[i]
            joint_positions = get_annotated_dictionary(joint_positions, 's21')
            # import ipdb; ipdb.set_trace()
            baseline_positions, baseline_out = baseline_forward_kinematics(
                angles=seq[i]
            )
            baseline_positions = convert_baseline_representation(baseline_positions)

            position_list.append(joint_positions)
            baseline_position_list.append(baseline_positions)

        import ipdb; ipdb.set_trace()
        # visualize_skeleton(position_list, H36M_SKELETON_STRUCTURE, title_text = title_info)
        time.sleep(3)
        # visualize_skeleton(baseline_position_list, H36M_SKELETON_STRUCTURE, title_text = title_info)
        compare_skeleton(
            baseline_position_list,
            position_list,
            H36M_SKELETON_STRUCTURE,
            H36M_NON_REDUNDANT_SKELETON_STRUCTURE
        )

    print("Done!")