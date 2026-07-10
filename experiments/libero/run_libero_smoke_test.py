import os

import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main() -> None:
    benchmark_dict = benchmark.get_benchmark_dict()

    task_suite_name = "libero_spatial"
    task_suite = benchmark_dict[task_suite_name]()

    task_id = 0
    task = task_suite.get_task(task_id)

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )

    print("=" * 70)
    print(f"Task suite : {task_suite_name}")
    print(f"Task name  : {task.name}")
    print(f"Instruction: {task.language}")
    print(f"BDDL file  : {task_bddl_file}")
    print("=" * 70)

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
    }

    env = None

    try:
        env = OffScreenRenderEnv(**env_args)
        env.seed(0)

        obs = env.reset()

        init_states = task_suite.get_task_init_states(task_id)
        print(f"Initial states: {len(init_states)}")

        obs = env.set_init_state(init_states[0])

        print("\nObservation keys:")

        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{key}: type={type(value).__name__}")

        dummy_action = np.zeros(7, dtype=np.float32)

        print("\nRunning dummy actions...")

        for step in range(10):
            obs, reward, done, info = env.step(dummy_action)

            print(
                f"step={step}, "
                f"reward={reward}, "
                f"done={done}"
            )

        print("\nLIBERO environment smoke test 성공")

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()