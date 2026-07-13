# 목표: 공개된 VLA 모델이 자연어 명령과 카메라 영상을 입력받아 LIBERO task에서 로봇 action을 생성하고, 실제로 task를 성공하는 과정을 직접 실행해보는 것
# 목적: VLA 모델을 직접 재현하고 평가하기 전에, 로봇 task·observation·action·dataset·평가 방식이 어떻게 연결되는지 실제 코드로 확인하는 기반 실험


## 배경
* TinyVLA의 reproduce를 위해 설치와 실행 환경 구성을 진행하였고, 패키지 설치, CUDA 연결, 학습 코드 진입까지는 확인하였습니다.
* 그러나 실제 학습을 위해서는 pretrained VLM checkpoint, 로봇 demonstration dataset, 데이터 경로 수정, 8 GPU 기준 학습 스크립트 수정 등이 추가로 필요하였습니다.
* 또한 GTX 1660 SUPER 6GB 환경에서는 TinyVLA 학습을 그대로 재현하기 어렵다고 판단하였습니다.
* 현재 GPU 환경에서 TinyVLA를 그대로 재현하기에는 부담이 크다고 판단하여, 먼저 LIBERO benchmark를 통해 VLA의 task, observation, action, dataset, 평가 구조를 실제 코드로 확인하고자 하였습니다.

## Progress

- [x] LIBERO 설치
- [x] LIBERO benchmark task 확인
- [x] MuJoCo simulation 실행
- [x] Observation 구조 확인
- [x] 7D dummy action 입력
- [x] Simulation step 실행
- [x] Camera observation 저장
- [x] LIBERO demonstration dataset 분석
- [ ] Baseline policy 실행
- [ ] OpenVLA-OFT evaluation
- [ ] RIPT-VLA 분석

## 단계별 목표

### 1단계: LIBERO 환경 실행 — 완료

LIBERO의 기본 실행 환경이 정상적으로 동작하는지 확인하였습니다.

* benchmark task 불러오기
* MuJoCo 기반 시뮬레이션 환경 생성
* observation 항목 및 shape 확인
* 7차원 action 입력
* simulation step 정상 실행 확인

이를 통해 LIBERO의 task, 초기 상태, observation, action이 실제 환경에서 정상적으로 연결되는 것을 확인하였습니다.

### 2단계: Demonstration Dataset 이해 — 완료

사람이 로봇을 조작하여 task를 성공한 demonstration data가 어떤 구조로 저장되어 있는지 확인합니다.

주요 확인 항목은 다음과 같습니다.

* 카메라 이미지
* 로봇 상태
* 객체 상태
* 각 시점의 action
* trajectory 길이
* episode 및 task 정보

이 단계의 목적은 TinyVLA 실행 과정에서 필요했던 HDF5 기반 robot demonstration dataset의 실제 구조를 이해하는 것입니다.

### 3단계: 정책 모델 실행 - 다음 작업

처음부터 모델을 학습하지 않고, 가능하면 공개된 pretrained 또는 fine-tuned checkpoint를 활용하여 정책 모델의 추론 과정을 확인합니다.

```text
Observation
→ Pretrained Policy
→ Predicted Action
→ LIBERO Environment
```

로컬 GPU 환경에서는 비교적 가벼운 LIBERO baseline이나 짧은 smoke test를 우선 수행합니다.

### 4단계: OpenVLA-OFT 평가

충분한 GPU 환경을 확보한 뒤 OpenVLA-OFT checkpoint를 LIBERO task에 적용합니다.

```text
OpenVLA-OFT Checkpoint
→ LIBERO Task 실행
→ Action 생성
→ Task 성공 여부 확인
→ Success Rate 평가
```

이를 통해 공개된 VLA 모델이 자연어 명령과 카메라 영상을 기반으로 실제 로봇 action을 생성하는 전체 평가 과정을 확인합니다.

### 5단계: RIPT-VLA 분석

OpenVLA-OFT의 fine-tuning 및 evaluation 구조를 이해한 뒤, RIPT-VLA가 추가한 interactive post-training 방법을 분석합니다.

주요 비교 항목은 다음과 같습니다.

* 학습 데이터 구성
* reward 또는 feedback 활용 방식
* post-training 절차
* LIBERO 평가 방식
* OpenVLA-OFT 대비 추가된 요소와 성능 변화


## 1. LIBERO 실행 

### 지금까지 한 작업
LIBERO 설치
libero_spatial task 목록 확인
자연어 명령과 BDDL task 파일 확인
MuJoCo 시뮬레이션 환경 생성
로봇 관절, gripper, 객체 위치, 카메라 이미지 등의 observation 확인
7차원 dummy action을 입력해 환경이 정상적으로 step 되는지 확인

