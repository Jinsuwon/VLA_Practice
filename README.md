## 목표: 공개된 VLA 모델이 자연어 명령과 카메라 영상을 입력받아 LIBERO task에서 로봇 action을 생성하고, 실제로 task를 성공하는 과정을 직접 실행해보는 것
## 목적: VLA 모델을 직접 재현하고 평가하기 전에, 로봇 task·observation·action·dataset·평가 방식이 어떻게 연결되는지 실제 코드로 확인하는 기반 실험


## 배경
* TinyVLA의 reproduce를 위해 설치와 실행 환경 구성을 진행하였고, 패키지 설치, CUDA 연결, 학습 코드 진입까지는 확인하였습니다.
* 그러나 실제 학습을 위해서는 pretrained VLM checkpoint, 로봇 demonstration dataset, 데이터 경로 수정, 8 GPU 기준 학습 스크립트 수정 등이 추가로 필요하였습니다.
* 또한 GTX 1660 SUPER 6GB 환경에서는 TinyVLA 학습을 그대로 재현하기 어렵다고 판단하였습니다.
* 현재 GPU 환경에서 TinyVLA를 그대로 재현하기에는 부담이 크다고 판단하여, 먼저 LIBERO benchmark를 통해 VLA의 task, observation, action, dataset, 평가 구조를 실제 코드로 확인하고자 하였습니다.

## Progress
<details>
<summary><b>환경 및 Task 확인</b></summary>

- [v] LIBERO 설치
- [v] LIBERO benchmark suite 확인
- [v] `libero_spatial` task 목록 확인
- [v] 자연어 task instruction 확인
- [v] BDDL task 파일 확인
- [v] MuJoCo simulation 환경 생성
- [v] initial state 불러오기
- [v] observation key 및 shape 확인
- [v] 7차원 dummy action 입력
- [v] simulation step 실행
- [v] camera observation 저장

</details>

<details>
<summary><b>Demonstration Dataset 확인</b></summary>

- [v] LIBERO demonstration HDF5 파일 확인
- [v] demonstration 개수 확인
- [v] trajectory 길이 확인
- [v] RGB image shape 확인
- [v] joint state shape 확인
- [v] gripper state shape 확인
- [v] 7차원 action shape 확인
- [v] timestep 단위 데이터 구조 확인
- [v] PyTorch Dataset 작성
- [v] DataLoader 작성
- [v] 실제 batch shape 확인

</details>

<details>
<summary><b>Lightweight Behavior Cloning Policy</b></summary>

- [v] CNN 기반 image encoder 작성
- [v] MLP 기반 robot state encoder 작성
- [v] 7차원 action head 작성
- [v] dummy input으로 model output shape 확인
- [v] train/validation demonstration 분리
- [v] MSE loss 기반 Behavior Cloning 학습
- [v] train loss 감소 확인
- [v] validation loss 감소 확인
- [v] best checkpoint 저장
- [v] 정답 action과 predicted action 비교
- [v] action dimension별 오차 분석

</details>

<details>
<summary><b>LIBERO Rollout 및 분석</b></summary>

- [v] 학습된 checkpoint 불러오기
- [v] LIBERO environment와 policy 연결
- [v] closed-loop rollout 실행
- [v] 300 timestep action 생성
- [v] rollout 결과 MP4 저장
- [v] 영상으로 로봇 동작 확인
- [v] 실패 구간 분석
- [v] bowl, plate, end-effector 위치 출력
- [v] bowl-plate XY 거리 계산
- [v] bowl-plate Z 차이 계산
- [v] release 실패 원인 확인
- [v] 정답 좌표 기반 release 조건 추가
- [v] 규칙 기반 개입 후 task success 확인
- [v] 학습 정책과 규칙 기반 정책의 차이 정리

</details>

<details>
<summary><b>이후 작업</b></summary>

- [ ] 여러 frame 또는 이전 action을 입력에 포함해 release 시점 판단 개선
- [ ] 단일 agentview 외에 wrist camera를 추가해 근접 조작 정보 보완

</details>


## 단계별 목표
<details>
<summary><b> 1단계: LIBERO 환경 실행 — 완료 </b></summary>

### 목표

LIBERO의 기본 실행 환경이 정상적으로 동작하는지 확인함.

### 수행한 작업

- benchmark suite 불러오기
- `libero_spatial` task 확인
- 자연어 명령 확인
- BDDL task 파일 확인
- MuJoCo simulation 환경 생성
- initial state 불러오기
- observation key와 shape 확인
- 7차원 dummy action 입력
- simulation step 정상 실행 확인
- agentview camera image 저장

### 확인한 observation 예시

```text
robot0_joint_pos
robot0_joint_vel
robot0_eef_pos
robot0_eef_quat
robot0_gripper_qpos
agentview_image
robot0_eye_in_hand_image
object-state
```

### 결과
LIBERO의 task, 초기 상태, observation, action, simulator가 정상적으로 연결되는 것을 확인하였음.
</details>

<details>
<summary><b> 2단계: Demonstration Dataset 이해 — 완료 </b></summary>

### 목표

사람이 로봇을 조작하여 task를 성공한 demonstration data가 어떤 구조로 저장되는지 확인하였음.

### 사용한 Task

```text
pick up the black bowl between the plate and the ramekin
and place it on the plate
```

### 사용한 Dataset

```text
pick_up_the_black_bowl_between_the_plate_and_the_ramekin_
and_place_it_on_the_plate_demo.hdf5
```

### 확인 결과

```text
demonstration 수: 50
전체 timestep 수: 5,068
demo_0 길이: 98
```

각 timestep에는 다음 데이터가 포함되어 있었음.

```text
agentview RGB image: (128, 128, 3)
joint state:         (7,)
gripper state:       (2,)
action:              (7,)
```

7차원 action은 다음과 같이 구성되어 있었음.

```text
[Δx, Δy, Δz, rotation 3개, gripper]
```

### 추가 작업

HDF5 데이터를 학습에 사용할 수 있도록 PyTorch Dataset과 DataLoader를 작성하였음.

```text
images:  [32, 3, 128, 128]
states:  [32, 9]
actions: [32, 7]
```

### 결과

TinyVLA 실행 과정에서 필요했던 HDF5 기반 robot demonstration dataset이 실제로 어떻게 구성되는지 확인하였음.

또한 demonstration을 모델 학습용 입력과 정답 action으로 변환하는 과정을 직접 구현하였음.


</details>

<details>
<summary><b> 3단계: Lightweight Behavior Cloning Policy 학습 — 완료</b></summary>

### 목표

대형 VLA 모델을 바로 사용하기 전에, 현재 GPU에서 실행 가능한 작은 정책 모델을 직접 작성하고 demonstration으로 학습하였음.

### 모델 구조

```text
Agentview RGB image
→ CNN image encoder
→ image feature

Joint state + gripper state
→ MLP state encoder
→ state feature

image feature + state feature
→ action head
→ 7D action
```

### 모델 규모

```text
총 파라미터 수: 46,855
```

### 학습 설정

```text
Train demos: demo_0 ~ demo_39
Validation demos: demo_40 ~ demo_49

Train samples: 4,130
Validation samples: 938

Batch size: 32
Optimizer: Adam
Learning rate: 0.001
Loss: MSE
Epochs: 10
```

### 학습 결과

```text
Train loss
0.208877 → 0.093428

Validation loss
0.193391 → 0.087726
```

Best checkpoint는 다음 위치에 저장하였음.

```text
outputs/libero/lightweight_policy_best.pt
```

### Offline Action 평가

Validation dataset에서 action dimension별 평균 절대 오차를 계산하였음.

```text
delta_x:    0.278847
delta_y:    0.101506
delta_z:    0.317710

rotation_1: 0.027457
rotation_2: 0.048054
rotation_3: 0.034005

gripper:    0.321649
```

### 분석

회전 action은 상대적으로 오차가 작았지만, `delta_x`, `delta_z`, `gripper`는 오차가 크게 나타났음.

특히 물체를 집고 내려놓는 데 중요한 높이 제어와 gripper 제어가 현재 모델의 주요 약점으로 확인되었음.

### 결과

LIBERO demonstration이 실제 policy 학습으로 연결되는 전체 Behavior Cloning pipeline을 구현하였음.

다만 이 모델은 언어 입력과 pretrained VLM을 사용하지 않으므로 VLA 모델은 아님.


</details>

<details>
<summary><b> 4단계: LIBERO Closed-loop Rollout — 완료 </b></summary>



### 목표

학습한 lightweight policy를 LIBERO simulator에 연결하고, 실제로 action을 생성하여 task를 수행하는지 확인하였음.

### 실행 구조

```text
Current image + robot state
→ Lightweight Policy
→ Predicted 7D action
→ LIBERO environment
→ Next observation
→ 반복
```

### 수행한 작업

- 학습된 checkpoint 불러오기
- LIBERO task environment 생성
- 현재 observation 불러오기
- 학습 과정과 동일한 이미지 및 robot state 전처리
- 모델을 이용한 7차원 action 예측
- 예측 action을 `[-1, 1]` 범위로 제한
- 예측 action을 LIBERO environment에 입력
- 새로운 observation을 다시 모델 입력으로 사용
- 최대 300 timestep의 closed-loop rollout 실행
- 매 timestep의 task success 여부 확인
- rollout 중 agentview image 저장
- 저장한 frame을 MP4 영상으로 변환

### 사용한 Checkpoint

```text
outputs/libero/lightweight_policy_best.pt
```

### 실행 방식

```text
현재 카메라 이미지 1장
+
현재 joint state와 gripper state
→ 현재 timestep의 7차원 action 1개 예측
→ simulator 실행
→ 다음 observation 획득
→ 동일 과정 반복
```

현재 정책은 이전 frame이나 이전 action을 직접 입력받지 않으며, 매 timestep의 현재 observation만을 이용하여 action을 예측함.

### 최초 Rollout 결과

```text
maximum rollout steps: 300
success: False
```

Checkpoint loading, observation preprocessing, action prediction, environment step, success 판정은 모두 정상적으로 실행되었으나 최종 task 수행에는 실패하였음.

### Rollout 영상 저장

Rollout의 동작을 직접 확인하기 위해 매 timestep의 `agentview_image`를 저장한 뒤 MP4 영상으로 생성하였음.

```text
outputs/libero/rollout_lightweight_policy.mp4
```

### 영상 분석 결과

학습 정책은 다음 동작을 어느 정도 수행하였음.

```text
1. 검은 그릇으로 접근
2. 그리퍼를 그릇 근처로 이동
3. 그릇 집기
4. 그릇 들어 올리기
5. 접시 방향으로 운반
6. 접시 위로 하강
```

그러나 다음 동작으로 안정적으로 전환하지 못하였음.

```text
7. 그리퍼를 열어 그릇을 접시 위에 놓기
```

그릇을 접시 근처까지 운반한 이후 적절한 시점에 gripper를 열지 못하였으며, 이후 비슷한 action을 반복하거나 그릇을 접시에서 다시 멀어지게 하였음.

### 결과

학습된 lightweight policy를 LIBERO simulator에 연결하여 다음 closed-loop pipeline을 직접 구현하였음.

```text
Observation
→ Policy
→ Predicted Action
→ Environment Step
→ Next Observation
```

Offline evaluation에서 loss가 감소하고 predicted action이 정답 action과 어느 정도 유사하더라도, 여러 timestep의 오차가 누적되는 closed-loop rollout에서는 task success가 보장되지 않음을 확인하였음.

또한 현재 단일 frame 기반 정책이 접근, 집기, 운반, 하강 동작은 일부 수행할 수 있으나, task 진행 상태를 판단하고 release 단계로 전환하는 능력에는 한계가 있음을 확인하였음.

</details>

<details>
<summary><b> 5단계: 실패 원인 분석 및 규칙 기반 개입 — 완료 </b></summary>

### 목표

Closed-loop rollout의 실패 원인을 영상과 simulator state를 이용하여 분석하고, release 시점의 문제를 구체적으로 확인하였음.

### 확인한 Observation Key

LIBERO observation에서 다음 객체 및 로봇 위치 정보를 확인하였음.

```text
robot0_eef_pos
akita_black_bowl_1_pos
plate_1_pos
```

각 위치 정보는 다음과 같은 3차원 좌표로 구성되어 있었음.

```text
[x, y, z]
```

### 추가로 출력한 값

Rollout 과정에서 다음 값을 10 timestep마다 출력하였음.

```text
bowl position
plate position
bowl-plate XY distance
bowl-plate Z difference
```

Bowl과 plate 사이의 평면 거리는 다음과 같이 계산하였음.

```python
bowl_plate_xy_distance = np.linalg.norm(
    bowl_position[:2] - plate_position[:2]
)
```

높이 차이는 다음과 같이 계산하였음.

```python
bowl_plate_z_difference = (
    bowl_position[2] - plate_position[2]
)
```

### 주요 시점 분석

```text
Step 70
XY distance: 0.0568
Z difference: 0.0685

Step 80
XY distance: 0.0435
Z difference: 0.0094

Step 90
XY distance: 0.0435
Z difference: 0.0094

Step 100
XY distance: 0.0488
Z difference: 0.0185
```

Step 80~90에서 bowl이 plate와 평면상 약 4.35 cm 거리까지 접근하였고, 높이 차이도 약 0.94 cm까지 감소하였음.

영상과 좌표를 함께 확인한 결과, 이 구간이 bowl을 plate 위에 내려놓고 gripper를 열어야 하는 시점으로 판단되었음.

### 실패 원인

학습 정책은 bowl을 plate 근처까지 운반하고 하강하는 동작까지 수행하였으나, 적절한 시점에 gripper를 열지 못하였음.

이후 bowl을 다시 들어 올리거나 plate에서 멀어지게 하였으며, 최종적으로 task success 조건을 만족하지 못하였음.

현재 정책은 매 timestep마다 다음 입력만 사용함.

```text
현재 agentview image 1장
현재 joint state
현재 gripper state
```

따라서 다음과 같은 task 진행 정보를 명시적으로 사용하지 못함.

```text
bowl을 이미 집었는지 여부
현재 운반 중인지 여부
plate 위에 도달했는지 여부
현재 release 단계인지 여부
이전 frame과 이전 action
```

### 초기 Release 조건

처음에는 다음과 같이 비교적 넓은 높이 조건을 사용하였음.

```python
near_plate_z = bowl_plate_z_difference < 0.08
```

이 조건은 bowl이 plate보다 최대 8 cm 위에 있는 상태까지 포함하므로, 운반 및 하강 과정에서도 조건이 만족될 수 있었음.

따라서 release 시점을 정확하게 구분하기에는 범위가 지나치게 넓었음.

### 수정한 Release 조건

실제 rollout 좌표를 기준으로 높이 조건을 다음과 같이 수정하였음.

```python
near_plate_z = (
    0.0 <= bowl_plate_z_difference < 0.02
)

if near_plate_xy and near_plate_z:
    action[6] = -1.0
```

`action[6] = -1.0`은 gripper를 여는 명령으로 사용하였음.

수정된 조건은 다음 상황에서 작동하도록 설정하였음.

```text
bowl과 plate의 XY 거리가 5 cm 미만
bowl이 plate보다 0~2 cm 위에 위치
→ gripper 열기
```

### 규칙 기반 개입 결과

수정한 release 조건을 적용한 뒤 task success를 확인하였음.

최종 동작은 다음과 같이 구성되었음.

```text
학습 정책
→ 접근
→ 집기
→ 들어 올리기
→ 운반
→ 하강

규칙 기반 개입
→ bowl과 plate의 상대 좌표 확인
→ 조건 만족 시 gripper 열기
→ task success
```

### 결과

이번 task success는 학습 정책만으로 달성한 결과가 아님.

최종 시스템은 다음 요소를 결합한 방식임.

```text
Behavior Cloning Policy
+
Simulator가 제공하는 bowl과 plate의 정답 좌표
+
사람이 정의한 release 조건
```

따라서 이를 카메라 observation만으로 task 전체를 해결한 순수 vision-based policy의 성공으로 해석할 수 없음.

학습 정책은 접근, 집기, 운반, 하강까지 어느 정도 수행하였으나, release 단계로 전환해야 하는 시점을 안정적으로 판단하지 못하였음.

정답 객체 좌표를 이용한 규칙 기반 release 조건을 추가하자 task가 성공하였음.

이를 통해 현재 정책의 주요 병목이 전체 action 생성 실패보다는 다음 요소에 있음을 확인하였음.

```text
temporal context 부족
task phase 인식 부족
release 시점 판단 부족
성공 상태 판단 부족
```

규칙 기반 개입은 일반화된 해결책은 아니지만, 현재 모델에서 부족한 기능이 release 전환 판단이라는 점을 구체적으로 진단하는 데 의미가 있었음.