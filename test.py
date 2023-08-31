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
    if args.model:
        test_transformer()
    if args.comp:
        test_processing_functions()
    if args.penc:
        test_pose_encoding_decoding()
    if args.metrics:
        test_metrics()
    if args.visualizer:
        test_visualizer()


if __name__ == "__main__":
    main()
