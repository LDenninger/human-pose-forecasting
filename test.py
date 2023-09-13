"""
    File to run different tests of the project.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""
import argparse

from src.tests import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--penc",
        action="store_true",
        default=False,
        help="Test the joint encoding and decoding of the pose predictor",
    )
    parser.add_argument(
        "--data", action="store_true", default=False, help="Test the data loading "
    )
    parser.add_argument(
        "--data_vlp",
        action="store_true",
        default=False,
        help="Test the data loading of the VisionLab3DPose dataset",
    )
    parser.add_argument(
        "--sk32", action="store_true", default=False, help="Test the Skeleton32 model"
    )
    parser.add_argument(
        "--sk21", action="store_true", default=False, help="Test the reduced skeleton model"
    )
    parser.add_argument(
        "--sk16", action="store_true", default=False, help="Test the stacked hourglass model"
    )
    parser.add_argument(
        "--h36mfk", action="store_true", default=False, help="Test the h36m forward kinematics function"
    )
    parser.add_argument(
        "--vis_bl", action="store_true", default=False, help="Test the baseline visualization function"
    )
    parser.add_argument(
        "--model", action="store_true", default=False, help="Test the Skeleton32 model"
    )
    parser.add_argument(
        "--comp",
        action="store_true",
        default=False,
        help="Test the high-level computation functions of the model",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        default=False,
        help="Test the metrics of the model",
    )
    parser.add_argument(
        "--visualizer",
        action="store_true",
        default=False,
        help="Test the visualizer of the model",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    if args.data:
        test_h36m_data_loading()
    if args.data_vlp:
        test_vslab_data_loading()
    if args.sk32:
        test_skeleton32_model()
    if args.sk21:
        test_s21_skeleton()
    if args.h36mfk:
        test_h36m_forward_kinematics()
    if args.vis_bl:
        test_baseline_visualization()
    if args.model:
        test_transformer()
    if args.comp:
        test_processing_functions()
    if args.penc:
        test_pose_encoding_decoding()
    if args.metrics:
        test_distribution_metrics()
    if args.visualizer:
        test_visualizer()


if __name__ == "__main__":
    main()
