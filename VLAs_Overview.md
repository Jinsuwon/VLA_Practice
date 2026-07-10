# VLA Model Comparison: Methodology, Architecture, Strengths, and Limitations

## Compared Models

- OpenVLA
- TinyVLA
- Hi-Robot
- CoT-VLA

---

## Comparison Table

| Category | OpenVLA | TinyVLA | Hi-Robot | CoT-VLA |
|---|---|---|---|---|
| **Methodology** | • pre-trained VLM을 robot action token prediction policy로 fine-tuning<br><br>• continuous robot action을 256-bin discretization<br><br>• action token을 language token처럼 예측하도록 학습 | • large-scale robot data pretraining 없이, compact pretrained VLM을 robot policy로 fine-tuning<br><br>• VLM은 visual-language 정보를 인코딩하고, action generation은 diffusion policy head가 담당<br><br>• VLM 대부분은 freeze하고 LoRA fine-tuning을 적용하여 task adaptation 수행<br><br>• action을 token으로 직접 예측하기보다, diffusion 기반으로 continuous action chunk를 생성 | • high-level reasoning과 low-level execution을 분리한 hierarchical VLA 구조 사용<br><br>• high-level policy가 task를 reasoning하고 subtask/subgoal 수준으로 분해<br><br>• low-level policy가 각 subtask를 실제 robot action으로 실행<br><br>• synthetic data generation scheme을 통해 high-level reasoning 학습 데이터를 확장 | • visual chain-of-thought reasoning을 VLA에 도입<br><br>• 현재 observation과 instruction을 보고 future subgoal image를 먼저 생성<br><br>• 생성된 subgoal image를 조건으로 short action sequence/action chunk 생성<br><br>• action-less video dataset도 subgoal image generation 학습에 활용<br><br>• 실행 후 새 observation을 받아 다시 subgoal과 action을 생성하는 closed-loop 방식 |
| **Structure Overview** | • DINOv2 + SigLIP visual encoder가 image feature 인코딩<br><br>• Llama 기반 7B VLM 사용<br><br>• least-used 256 vocabulary token을 action token으로 재정의<br><br>• action token을 autoregressive하게 예측<br><br>• 예측된 action token을 robot action으로 변환 | • compact VLM backbone 사용<br><br>• visual-language encoder + diffusion policy decoder 구조<br><br>• image와 language instruction을 입력으로 받아 multimodal representation 생성<br><br>• diffusion policy head가 action chunk를 생성<br><br>• LoRA를 통해 VLM 일부만 효율적으로 fine-tuning | • hierarchical 구조: high-level policy + low-level policy<br><br>• high-level policy는 image-language input을 바탕으로 task-level reasoning 수행<br><br>• low-level policy는 PaliGemma 기반 VLM을 fine-tuning하여 manipulation action 수행<br><br>• robot observation과 target atomic command를 VLM에 입력하여, 해당 command를 유도했을 법한 complex prompt·user feedback·interjection을 synthetic하게 생성<br><br>• 전체 구조는 long-horizon instruction을 여러 단계의 executable behavior로 분해 | • VILA-U 기반 unified multimodal foundation model 사용<br><br>• text, image, action token을 함께 다루는 7B VLA 구조<br><br>• causal attention으로 text/image/subgoal image 생성<br><br>• full attention으로 action dimensions를 병렬 예측<br><br>• observation + instruction → subgoal image → action chunk → robot execution → new observation 반복 |
| **Strength** | • pretrained VLM의 시각-언어 지식을 robot policy에 활용<br><br>• DINOv2의 spatial feature와 SigLIP의 semantic feature를 함께 사용<br><br>• 다양한 embodiment와 task에 대한 generalization 목표<br><br>• LoRA fine-tuning을 통한 adaptation 가능<br><br>• int4 quantized inference를 통한 제한된 compute 환경에서 inference 가능 | • OpenVLA보다 경량 구조이므로 inference 속도와 학습 효율성이 좋음<br><br>• large-scale robot data pretraining에 덜 의존하여 data-efficient한 학습 가능<br><br>• diffusion policy head를 사용해 continuous action을 더 자연스럽게 생성 가능<br><br>• VLM 대부분을 freeze하고 LoRA를 사용하므로 제한된 compute 환경에서도 adaptation 가능<br><br>• compact 구조로 practical deployment에 유리 | • long-horizon task를 한 번에 action으로 예측하지 않고 단계적으로 분해 가능<br><br>• high-level reasoning과 low-level control을 분리하여 복잡한 instruction 처리에 유리<br><br>• synthetic data를 활용해 high-level reasoning 학습 데이터를 효율적으로 확장<br><br>• hierarchical decomposition을 통해 task planning 성격이 OpenVLA/TinyVLA보다 명확함<br><br>• reasoning과 execution을 역할별로 나누어 모델 구조 해석이 비교적 명확 | • action을 바로 예측하기 전에 visual subgoal을 생성하여 reasoning 과정을 명시화<br><br>• future image frame을 중간 reasoning step으로 사용해 temporal planning 성격 강화<br><br>• action annotation이 없는 video data도 visual reasoning 학습에 활용 가능<br><br>• subgoal image가 시각적으로 해석 가능해 모델이 무엇을 목표로 하는지 확인하기 쉬움<br><br>• hybrid attention과 action chunking을 통해 action prediction 성능 개선 |
| **Limitation** | • single-image observation 사용<br><br>• low control frequency<br><br>• underexplored design decisions 존재<br><br>• explicit reasoning, task planning, failure recovery를 직접 다루지 않음<br><br>• long-horizon task에서는 별도의 planner나 replanning module 필요 | • compact model이므로 대규모 VLA 대비 일반화 능력에는 한계 가능<br><br>• explicit reasoning이나 high-level task planning을 직접 다루는 구조는 아님<br><br>• long-horizon task에서는 별도의 planner나 hierarchical module이 필요할 수 있음<br><br>• diffusion policy head 성능이 action generation 품질에 크게 영향<br><br>• 복잡한 instruction reasoning보다는 manipulation policy 효율화에 초점이 있음 | • high-level policy의 subtask/subgoal 생성 오류가 low-level execution 실패로 이어질 수 있음<br><br>• high-level reasoning과 low-level execution 사이의 연결 품질에 성능이 크게 의존<br><br>• synthetic data 품질이 reasoning 성능에 영향을 줌<br><br>• low-level policy 자체의 manipulation 능력 한계를 넘어서기는 어려움<br><br>• end-to-end 단순 구조보다 시스템 복잡도가 높음 | • subgoal image 생성 품질이 action generation 성능에 직접적인 영향을 줌<br><br>• image generation 단계를 추가하므로 일반 VLA보다 시스템 복잡도와 계산 비용이 커질 수 있음<br><br>• 생성된 subgoal이 물리적으로 실행 가능하지 않으면 downstream action도 실패할 수 있음<br><br>• explicit language-level task planning이나 symbolic planning을 직접 수행하는 구조는 아님<br><br>• long-horizon task에서는 여러 번의 subgoal generation 과정에서 오류가 누적될 수 있음 |

---

## Short Summary

| Model | Core Idea |
|---|---|
| **OpenVLA** | Pretrained VLM을 robot action token prediction policy로 fine-tuning한 end-to-end VLA |
| **TinyVLA** | Compact VLM과 diffusion policy head를 결합해 빠르고 data-efficient한 VLA를 목표로 함 |
| **Hi-Robot** | High-level reasoning과 low-level execution을 분리한 hierarchical VLA |
| **CoT-VLA** | Future subgoal image를 중간 reasoning step으로 생성하는 visual chain-of-thought VLA |

---

## Takeaway

OpenVLA와 TinyVLA는 주로 **VLA를 action policy로 어떻게 만들 것인가**에 초점을 둔다.

Hi-Robot과 CoT-VLA는 action을 바로 예측하기 전에 중간 단계를 넣는다.  
Hi-Robot은 **language command-level hierarchical decomposition**을 사용하고, CoT-VLA는 **future subgoal image 기반 visual reasoning**을 사용한다.

따라서 reasoning이나 task planning 관점에서는 OpenVLA/TinyVLA보다 Hi-Robot과 CoT-VLA가 더 직접적으로 관련된다.