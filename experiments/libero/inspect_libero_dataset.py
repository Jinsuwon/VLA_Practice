from pathlib import Path

import h5py


DATASET_ROOT = Path(
    "/home/jinsuwon/VLA_Practice/repos/LIBERO/libero/datasets"
)


def print_hdf5_structure(group: h5py.Group, indent: int = 0) -> None:
    """HDF5 내부의 group과 dataset 구조를 재귀적으로 출력합니다."""
    prefix = "  " * indent

    for key, item in group.items():
        if isinstance(item, h5py.Group):
            print(f"{prefix}[Group] {key}/")
            print_hdf5_structure(item, indent + 1)

        elif isinstance(item, h5py.Dataset):
            print(
                f"{prefix}[Dataset] {key}: "
                f"shape={item.shape}, dtype={item.dtype}"
            )


def find_first_hdf5_file(root: Path) -> Path:
    """데이터셋 폴더에서 첫 번째 HDF5 파일을 찾습니다."""
    files = sorted(list(root.rglob("*.hdf5")) + list(root.rglob("*.h5")))

    if not files:
        raise FileNotFoundError(
            f"HDF5 파일을 찾지 못했습니다: {root}\n"
            "먼저 LIBERO dataset을 다운로드했는지 확인하십시오."
        )

    return files[0]


def print_attributes(obj: h5py.Group, title: str) -> None:
    """HDF5 객체에 저장된 attribute를 출력합니다."""
    print(f"\n{title}")

    if not obj.attrs:
        print("  저장된 attribute가 없습니다.")
        return

    for key, value in obj.attrs.items():
        print(f"  {key}: {value}")


def main() -> None:
    dataset_path = find_first_hdf5_file(DATASET_ROOT)

    print("=" * 80)
    print("Dataset file")
    print(dataset_path)
    print("=" * 80)

    with h5py.File(dataset_path, "r") as file:
        print_attributes(file, "Root attributes")

        print("\nHDF5 structure")
        print_hdf5_structure(file)

        # LIBERO 데이터셋은 일반적으로 data/demo_* 구조를 사용합니다.
        if "data" not in file:
            print("\n'data' group이 없습니다.")
            return

        data_group = file["data"]
        print_attributes(data_group, "data group attributes")

        demo_names = sorted(data_group.keys())

        print(f"\nNumber of demonstrations: {len(demo_names)}")

        if not demo_names:
            print("저장된 demonstration이 없습니다.")
            return

        first_demo_name = demo_names[0]
        demo = data_group[first_demo_name]

        print(f"Selected demonstration: {first_demo_name}")
        print_attributes(demo, f"{first_demo_name} attributes")

        print("\nSelected demonstration structure")
        print_hdf5_structure(demo)

        if "actions" in demo:
            actions = demo["actions"]

            print("\nAction information")
            print(f"  shape: {actions.shape}")
            print(f"  dtype: {actions.dtype}")

            if len(actions.shape) >= 1:
                print(f"  trajectory length: {actions.shape[0]}")

            if actions.shape[0] > 0:
                print(f"  first action: {actions[0]}")

        if "obs" in demo:
            obs = demo["obs"]

            print("\nObservation information")

            for key, dataset in obs.items():
                if isinstance(dataset, h5py.Dataset):
                    print(
                        f"  {key}: "
                        f"shape={dataset.shape}, "
                        f"dtype={dataset.dtype}"
                    )


if __name__ == "__main__":
    main()