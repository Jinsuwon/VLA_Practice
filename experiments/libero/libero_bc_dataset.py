from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


DATASET_ROOT = Path(
    "/home/jinsuwon/VLA_Practice/repos/LIBERO/libero/datasets/libero_spatial"
)

TASK_KEYWORD = "between_the_plate_and_the_ramekin"


def find_dataset_file() -> Path:
    files = sorted(DATASET_ROOT.glob("*.hdf5"))

    matching_files = [
        path for path in files
        if TASK_KEYWORD in path.name
    ]

    if not matching_files:
        raise FileNotFoundError(
            f"HDF5 파일을 찾지 못했습니다: {TASK_KEYWORD}"
        )

    return matching_files[0]


class LiberoBCDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        demo_names: List[str],
    ):
        self.dataset_path = dataset_path
        self.demo_names = demo_names

        # 각 샘플이 어느 demo의 몇 번째 timestep인지 저장
        self.sample_index: List[Tuple[str, int]] = []

        with h5py.File(self.dataset_path, "r") as file:
            data_group = file["data"]

            for demo_name in self.demo_names:
                demo = data_group[demo_name]
                trajectory_length = demo["actions"].shape[0]

                for timestep in range(trajectory_length):
                    self.sample_index.append(
                        (demo_name, timestep)
                    )

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int):
        demo_name, timestep = self.sample_index[index]

        with h5py.File(self.dataset_path, "r") as file:
            demo = file["data"][demo_name]

            image = demo["obs"]["agentview_rgb"][timestep]
            image = np.flipud(image).copy()

            joint_state = demo["obs"]["joint_states"][timestep]
            gripper_state = demo["obs"]["gripper_states"][timestep]
            action = demo["actions"][timestep]

        robot_state = np.concatenate(
            [joint_state, gripper_state],
            axis=0,
        )

        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor.float() / 255.0

        robot_state_tensor = torch.from_numpy(
            robot_state.astype(np.float32)
        )

        action_tensor = torch.from_numpy(
            action.astype(np.float32)
        )

        return {
            "image": image_tensor,
            "robot_state": robot_state_tensor,
            "action": action_tensor,
            "demo_name": demo_name,
            "timestep": timestep,
        }


def main() -> None:
    dataset_path = find_dataset_file()

    # 우선 50개 demonstration 전체 사용
    demo_names = [
        f"demo_{index}"
        for index in range(50)
    ]

    dataset = LiberoBCDataset(
        dataset_path=dataset_path,
        demo_names=demo_names,
    )

    print("=" * 70)
    print("Dataset file")
    print(dataset_path)
    print("=" * 70)

    print(f"\nDemonstration count: {len(demo_names)}")
    print(f"Total sample count: {len(dataset)}")

    first_sample = dataset[0]

    print("\nFirst sample")
    print(f"demo name: {first_sample['demo_name']}")
    print(f"timestep: {first_sample['timestep']}")
    print(f"image shape: {first_sample['image'].shape}")
    print(
        f"robot state shape: "
        f"{first_sample['robot_state'].shape}"
    )
    print(f"action shape: {first_sample['action'].shape}")

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    first_batch = next(iter(dataloader))

    print("\nFirst batch")
    print(f"image batch shape: {first_batch['image'].shape}")
    print(
        f"robot state batch shape: "
        f"{first_batch['robot_state'].shape}"
    )
    print(
        f"action batch shape: "
        f"{first_batch['action'].shape}"
    )


if __name__ == "__main__":
    main()