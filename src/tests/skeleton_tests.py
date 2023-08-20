import torch
from ..data_utils import SkeletonModel32, H36MDataset, H36M_SKELETON_STRUCTURE, baseline_forward_kinematics, convert_baseline_representation, expmap2rotmat, axis_angle_to_matrix
from ..visualization import visualize_skeleton, compare_skeleton
import time

def test_skeleton32_model():

    skeleton = SkeletonModel32()

    dataset = H36MDataset(sequence_length=40, down_sampling_factor=2)

    for seq, label in dataset:
        position_list = []
        baseline_position_list = []

        for i in range(seq.shape[0]):
            #import ipdb; ipdb.set_trace()
            skeleton(seq[i])
            joint_positions = skeleton.get_joint_positions(incl_names=True)
            #import ipdb; ipdb.set_trace()
            baseline_positions, baseline_out = baseline_forward_kinematics(angles=seq[i])
            baseline_positions = convert_baseline_representation(baseline_positions)

            position_list.append(joint_positions)
            baseline_position_list.append(baseline_positions)
        import ipdb; ipdb.set_trace()
        title_info = f" Person: {label['person_id']}, Action: {label['action_str']}, SubAction: {label['sub_action_id']}"
        time.sleep(1)
        #visualize_skeleton(position_list, H36M_SKELETON_STRUCTURE, title_text = title_info)
        time.sleep(3)
        #visualize_skeleton(baseline_position_list, H36M_SKELETON_STRUCTURE, title_text = title_info)
        compare_skeleton( baseline_position_list, position_list, H36M_SKELETON_STRUCTURE, title_text=title_info)
    
    print("Done!")