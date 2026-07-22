from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from libero_bc_dataset import LiberoBCDataset, find_dataset_file
from lightweight_policy import LightweightPolicy


OUTPUT_DIR = Path(
    "/home/jinsuwon/VLA_Practice/outputs/libero"
)

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10


def run_validation(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.eval()

    total_loss = 0.0
    batch_count = 0

    # validation에서는 학습하지 않으므로 gradient 계산을 끔
    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device)
            robot_state = batch["robot_state"].to(device)
            target_action = batch["action"].to(device)

            predicted_action = model(
                image,
                robot_state,
            )

            loss = loss_function(
                predicted_action,
                target_action,
            )

            total_loss += loss.item()
            batch_count += 1

    return total_loss / batch_count


def main() -> None:
    dataset_path = find_dataset_file()

    # 같은 demonstration이 train과 validation에 동시에 들어가지 않도록 분리
    train_demo_names = [
        f"demo_{index}"
        for index in range(40)
    ]

    validation_demo_names = [
        f"demo_{index}"
        for index in range(40, 50)
    ]

    train_dataset = LiberoBCDataset(
        dataset_path=dataset_path,
        demo_names=train_demo_names,
    )

    validation_dataset = LiberoBCDataset(
        dataset_path=dataset_path,
        demo_names=validation_demo_names,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = LightweightPolicy().to(device)

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("=" * 70)
    print("Lightweight policy training")
    print("=" * 70)
    print(f"device: {device}")
    print(f"dataset file: {dataset_path}")
    print(f"train demonstration count: {len(train_demo_names)}")
    print(f"validation demonstration count: {len(validation_demo_names)}")
    print(f"train sample count: {len(train_dataset)}")
    print(f"validation sample count: {len(validation_dataset)}")
    print(f"batch size: {BATCH_SIZE}")
    print(f"epoch count: {EPOCHS}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_validation_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_train_loss = 0.0
        train_batch_count = 0

        for batch in train_dataloader:
            image = batch["image"].to(device)
            robot_state = batch["robot_state"].to(device)
            target_action = batch["action"].to(device)

            # 이전 batch의 gradient 제거
            optimizer.zero_grad()

            # action 예측
            predicted_action = model(
                image,
                robot_state,
            )

            # 예측값과 정답값 차이 계산
            loss = loss_function(
                predicted_action,
                target_action,
            )

            # gradient 계산
            loss.backward()

            # 모델 parameter 업데이트
            optimizer.step()

            total_train_loss += loss.item()
            train_batch_count += 1

        average_train_loss = (
            total_train_loss / train_batch_count
        )

        average_validation_loss = run_validation(
            model=model,
            dataloader=validation_dataloader,
            loss_function=loss_function,
            device=device,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss: {average_train_loss:.6f} | "
            f"validation loss: {average_validation_loss:.6f}"
        )

        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss

            checkpoint_path = (
                OUTPUT_DIR
                / "lightweight_policy_best.pt"
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_loss": average_validation_loss,
                },
                checkpoint_path,
            )

            print(
                f"  saved best checkpoint: "
                f"{checkpoint_path}"
            )

    print("\nTraining finished")
    print(
        f"best validation loss: "
        f"{best_validation_loss:.6f}"
    )


if __name__ == "__main__":
    main()