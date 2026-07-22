from pathlib import Path

import torch
from torch.utils.data import DataLoader

from libero_bc_dataset import LiberoBCDataset, find_dataset_file
from lightweight_policy import LightweightPolicy


CHECKPOINT_PATH = Path(
    "/home/jinsuwon/VLA_Practice/outputs/libero/"
    "lightweight_policy_best.pt"
)


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset_path = find_dataset_file()

    validation_demo_names = [
        f"demo_{index}"
        for index in range(40, 50)
    ]

    validation_dataset = LiberoBCDataset(
        dataset_path=dataset_path,
        demo_names=validation_demo_names,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    model = LightweightPolicy().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    total_absolute_error = torch.zeros(
        7,
        device=device,
    )

    total_sample_count = 0

    first_batch_prediction = None
    first_batch_target = None

    with torch.no_grad():
        for batch in validation_dataloader:
            image = batch["image"].to(device)
            robot_state = batch["robot_state"].to(device)
            target_action = batch["action"].to(device)

            predicted_action = model(
                image,
                robot_state,
            )

            absolute_error = torch.abs(
                predicted_action - target_action
            )

            total_absolute_error += absolute_error.sum(dim=0)
            total_sample_count += target_action.shape[0]

            if first_batch_prediction is None:
                first_batch_prediction = (
                    predicted_action.cpu()
                )
                first_batch_target = target_action.cpu()

    mean_absolute_error = (
        total_absolute_error / total_sample_count
    )

    print("=" * 70)
    print("Lightweight policy evaluation")
    print("=" * 70)

    print(f"device: {device}")
    print(f"checkpoint epoch: {checkpoint['epoch']}")
    print(
        f"checkpoint validation loss: "
        f"{checkpoint['validation_loss']:.6f}"
    )
    print(f"validation sample count: {total_sample_count}")

    print("\nFirst five action comparisons")

    for index in range(5):
        print(f"\nSample {index}")
        print(
            "target:    ",
            first_batch_target[index],
        )
        print(
            "predicted: ",
            first_batch_prediction[index],
        )
        print(
            "error:     ",
            torch.abs(
                first_batch_prediction[index]
                - first_batch_target[index]
            ),
        )

    print("\nMean absolute error by action dimension")

    action_names = [
        "delta_x",
        "delta_y",
        "delta_z",
        "rotation_1",
        "rotation_2",
        "rotation_3",
        "gripper",
    ]

    for name, error in zip(
        action_names,
        mean_absolute_error,
    ):
        print(f"{name:>12}: {error.item():.6f}")


if __name__ == "__main__":
    main()