import torch
import torch.nn as nn


class LightweightPolicy(nn.Module):
    def __init__(
        self,
        robot_state_dim: int = 9,
        action_dim: int = 7,
    ):
        super().__init__()

        # 이미지 특징 추출
        self.image_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            # 이미지 크기와 무관하게 1×1로 압축
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # 로봇 상태 특징 추출
        self.state_encoder = nn.Sequential(
            nn.Linear(robot_state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # 이미지 특징과 로봇 상태 특징 결합
        self.action_head = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(
        self,
        image: torch.Tensor,
        robot_state: torch.Tensor,
    ) -> torch.Tensor:
        image_feature = self.image_encoder(image)

        # (batch, 64, 1, 1)
        # → (batch, 64)
        image_feature = image_feature.flatten(start_dim=1)

        state_feature = self.state_encoder(robot_state)

        combined_feature = torch.cat(
            [image_feature, state_feature],
            dim=1,
        )

        predicted_action = self.action_head(combined_feature)

        return predicted_action


def main() -> None:
    model = LightweightPolicy()

    # 임시 입력
    batch_size = 32

    dummy_image = torch.randn(
        batch_size,
        3,
        128,
        128,
    )

    dummy_robot_state = torch.randn(
        batch_size,
        9,
    )

    predicted_action = model(
        dummy_image,
        dummy_robot_state,
    )

    print("=" * 70)
    print("Lightweight policy test")
    print("=" * 70)

    print(f"image input shape: {dummy_image.shape}")
    print(
        f"robot state input shape: "
        f"{dummy_robot_state.shape}"
    )
    print(
        f"predicted action shape: "
        f"{predicted_action.shape}"
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"parameter count: {parameter_count:,}")


if __name__ == "__main__":
    main()