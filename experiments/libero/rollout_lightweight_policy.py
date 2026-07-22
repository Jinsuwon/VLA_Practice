import os
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path

from lightweight_policy import LightweightPolicy



CHECKPOINT_PATH = Path(
    "/home/jinsuwon/VLA_Practice/outputs/libero/"
    "lightweight_policy_best.pt"
)

BENCHMARK_NAME = "libero_spatial"
TASK_ID = 0
INIT_STATE_ID = 0

MAX_STEPS = 300
IMAGE_SIZE = 128

VIDEO_PATH = Path(
    "/home/jinsuwon/VLA_Practice/outputs/libero/"
    "rollout_lightweight_policy.mp4"
)

VIDEO_FPS = 20


def preprocess_observation(
    observation: dict,
    device: torch.device,
):
    # Simulator 카메라 이미지
    image = observation["agentview_image"]

    # HDF5 학습 데이터와 같은 방향으로 변환
    image = np.flipud(image).copy()

    # (H, W, C) → (C, H, W)
    image_tensor = torch.from_numpy(image)
    image_tensor = image_tensor.permute(2, 0, 1)

    # uint8 0~255 → float32 0~1
    image_tensor = image_tensor.float() / 255.0

    # batch 차원 추가
    # (3, 128, 128) → (1, 3, 128, 128)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    joint_state = observation["robot0_joint_pos"]
    gripper_state = observation["robot0_gripper_qpos"]

    robot_state = np.concatenate(
        [joint_state, gripper_state],
        axis=0,
    ).astype(np.float32)

    # (9,) → (1, 9)
    robot_state_tensor = (
        torch.from_numpy(robot_state)
        .unsqueeze(0)
        .to(device)
    )

    return image_tensor, robot_state_tensor


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("Lightweight policy rollout")
    print("=" * 70)
    print(f"device: {device}")

    # ---------------------------------------------------------
    # 1. 학습한 모델 불러오기
    # ---------------------------------------------------------
    model = LightweightPolicy().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(f"checkpoint epoch: {checkpoint['epoch']}")
    print(
        f"checkpoint validation loss: "
        f"{checkpoint['validation_loss']:.6f}"
    )

    # ---------------------------------------------------------
    # 2. LIBERO task 불러오기
    # ---------------------------------------------------------
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[BENCHMARK_NAME]()

    task = task_suite.get_task(TASK_ID)

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )

    print(f"benchmark: {BENCHMARK_NAME}")
    print(f"task id: {TASK_ID}")
    print(f"task language: {task.language}")
    print(f"BDDL file: {task_bddl_file}")

    # ---------------------------------------------------------
    # 3. Simulator 생성
    # ---------------------------------------------------------
    environment = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=IMAGE_SIZE,
        camera_widths=IMAGE_SIZE,
    )

    environment.seed(0)
    environment.reset()

    initial_states = task_suite.get_task_init_states(
        TASK_ID
    )

    observation = environment.set_init_state(
        initial_states[INIT_STATE_ID]
    )

    print("\nObservation keys")

    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape={value.shape}")
        else:
            print(f"{key}: {type(value)}")


    print(f"initial state id: {INIT_STATE_ID}")
    print(f"maximum rollout steps: {MAX_STEPS}")
    print("\nRollout started")

    frames = []
    success = False

    # ---------------------------------------------------------
    # 4. Closed-loop rollout
    # ---------------------------------------------------------
    for step in range(MAX_STEPS):
        frame = observation["agentview_image"]
        frame = np.flipud(frame).copy()
        frames.append(frame)

        image, robot_state = preprocess_observation(
            observation=observation,
            device=device,
        )

        with torch.no_grad():
            predicted_action = model(
                image,
                robot_state,
            )

        # (1, 7) → (7,)
        action = (
            predicted_action
            .squeeze(0)
            .cpu()
            .numpy()
        )
        bowl_position = observation["akita_black_bowl_1_pos"]
        plate_position = observation["plate_1_pos"]

        bowl_plate_xy_distance = np.linalg.norm(
            bowl_position[:2] - plate_position[:2]
        )

        bowl_plate_z_difference = (
            bowl_position[2] - plate_position[2]
        )

        near_plate_xy = bowl_plate_xy_distance < 0.05
        near_plate_z = bowl_plate_z_difference < 0.08

        if near_plate_xy and near_plate_z:
            action[6] = -1.0
            
        # LIBERO action 범위에 맞게 제한
        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        observation, reward, done, info = (
            environment.step(action)
        )

        success = bool(
            environment.check_success()
        )
        if step % 10 == 0:
            eef_position = observation["robot0_eef_pos"]
            bowl_position = observation["akita_black_bowl_1_pos"]
            plate_position = observation["plate_1_pos"]

            bowl_plate_xy_distance = np.linalg.norm(
                bowl_position[:2] - plate_position[:2]
            )

            bowl_plate_z_difference = (
                bowl_position[2] - plate_position[2]
            )

            print(
                f"step: {step:03d} | "
                f"reward: {float(reward):.4f} | "
                f"success: {success}"
            )

            print(
                f"  action: {np.round(action, 3)}"
            )

            print(
                f"  eef position:   "
                f"{np.round(eef_position, 3)}"
            )

            print(
                f"  bowl position:  "
                f"{np.round(bowl_position, 3)}"
            )

            print(
                f"  plate position: "
                f"{np.round(plate_position, 3)}"
            )

            print(
                f"  bowl-plate xy distance: "
                f"{bowl_plate_xy_distance:.4f}"
            )

            print(
                f"  bowl-plate z difference: "
                f"{bowl_plate_z_difference:.4f}"
            )

        if success:
            print(
                f"\nTask succeeded at step {step}"
            )
            break

        if done:
            print(
                f"\nEnvironment ended at step {step}"
            )
            break

    print("\nRollout finished")
    print(f"success: {success}")

    VIDEO_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    imageio.mimsave(
        VIDEO_PATH,
        frames,
        fps=VIDEO_FPS,
        codec="libx264",
    )

    print(f"video saved: {VIDEO_PATH}")
    environment.close()


if __name__ == "__main__":
    main()