from pathlib import Path
from PIL import Image

import h5py
import numpy as np


DATASET_ROOT = Path(
    "/home/jinsuwon/VLA_Practice/repos/LIBERO/libero/datasets/libero_spatial"
)

TASK_KEYWORD = "between_the_plate_and_the_ramekin"


def find_dataset_file() -> Path:
    files = sorted(DATASET_ROOT.glob("*.hdf5"))

    matching_files = [
        path for path in files if TASK_KEYWORD in path.name
    ]

    if not matching_files:
        raise FileNotFoundError(
            f"다음 task의 HDF5 파일을 찾지 못했습니다: {TASK_KEYWORD}"
        )

    return matching_files[0]


def main() -> None:
    dataset_path = find_dataset_file()

    print("=" * 70)
    print("Dataset file")
    print(dataset_path)
    print("=" * 70)

    with h5py.File(dataset_path, "r") as file:
        demo = file["data"]["demo_0"]

        timestep = 0

        image = demo["obs"]["agentview_rgb"][timestep]
        image = np.flipud(image).copy()
        joint_state = demo["obs"]["joint_states"][timestep]
        gripper_state = demo["obs"]["gripper_states"][timestep]
        action = demo["actions"][timestep]

        output_dir = Path("/home/jinsuwon/VLA_Practice/outputs/libero")
        output_dir.mkdir(parents=True, exist_ok=True)

        image_path = output_dir / "demo_0_timestep_0.png"
        Image.fromarray(image).save(image_path)

        print("\nSaved image")
        print(image_path)
        
        joint_state = demo["obs"]["joint_states"][timestep]
        gripper_state = demo["obs"]["gripper_states"][timestep]
        action = demo["actions"][timestep]

        robot_state = np.concatenate(
            [joint_state, gripper_state],
            axis=0,
        )

        print(f"\nSelected demo: demo_0")
        print(f"Selected timestep: {timestep}")

        print("\nImage")
        print(f"shape: {image.shape}")
        print(f"dtype: {image.dtype}")
        print(f"min pixel value: {image.min()}")
        print(f"max pixel value: {image.max()}")

        print("\nJoint state")
        print(f"shape: {joint_state.shape}")
        print(joint_state)

        print("\nGripper state")
        print(f"shape: {gripper_state.shape}")
        print(gripper_state)

        print("\nCombined robot state")
        print(f"shape: {robot_state.shape}")
        print(robot_state)

        print("\nTarget action")
        print(f"shape: {action.shape}")
        print(action)


if __name__ == "__main__":
    main()