# LIBERO public demos → N1.7 BC → offline-RL preparation

새 rollout 수집 없이 공개 demonstration으로 BC를 학습하는 실행기입니다.
기존 N1.7 환경과 base weight를 재사용합니다. **이 실행기는 BC 전용이며, DEAS critic을
N1.7에 이식하거나 SVF용 reward/종료 라벨을 생성하지 않습니다.** 현재 carrot 학습을
중단하거나 자동으로 LIBERO 학습으로 전환하지도 않습니다.

## Two independent GPU runs / checkpoint comparison

`--num-gpus 1 --gpu 0` / `--gpu 1`로 physical GPU를 각각 지정합니다.
실행기는 GPU UUID로 CUDA visibility를 고정하고 GPU별 파일 잠금과 점유 검사를
수행합니다. 다른 GPU의 학습은 허용하지만 같은 GPU의 기존 학습은 종료하지 않고
실행을 거부합니다. W&B run과 output은 서로 분리됩니다.

```bash
# 각각 별도 tmux에서 실행. --execute 없이는 명령 출력만 수행.
bash examples/LIBERO/finetune/run.sh \
  --profile qvgm-long --num-gpus 1 --gpu 0 --batch-per-gpu 32 \
  --steps 5000 --save-steps 5000 \
  --save-at-steps 400 500 1000 2000 3000 5000 \
  --output /raid/yoon/vla_finetune/outputs/libero-long-bc

bash examples/LIBERO/finetune/run.sh \
  --profile qvgm-unified --num-gpus 1 --gpu 1 --batch-per-gpu 32 \
  --steps 5000 --save-steps 5000 \
  --save-at-steps 500 1000 2000 3000 5000 \
  --output /raid/yoon/vla_finetune/outputs/libero-unified-bc
```

- `qvgm-long`: Long 10 tasks × 5 demos = 50 episodes, FP32 + Qwen SDPA.
- `qvgm-unified`: four suites, 40 tasks × 1 demo = 40 episodes, BF16 mixed precision.
- Preset source/budget caveats are recorded in `qvgm_presets.json`. The unified profile
  requires explicit `--steps`, because Q-VGM reuses a checkpoint rather than specifying
  that checkpoint's SFT update count.
- `--save-at-steps` adds exact milestones to the regular cadence. Above, each model
  trains once for 5,000 optimizer updates; the schedule is **not restarted** at milestones.
  Consequently a 400-step checkpoint from this 5,000-step LR schedule is not an exact
  reproduction of a separate 400-step-scheduled Q-VGM run.
- Latest optimizer/scheduler/RNG is retained for resume; earlier milestone model weights
  remain for evaluation. The final step is saved even if it is not a regular save step.
- Checkpoint choice should use fixed LIBERO evaluation initial states/seeds, not only BC
  loss. Evaluation and recurring monitoring are not automatically launched.

## DEAS의 실제 actor / critic 구조

확인한 DEAS commit: `1bfc8464c4d28574cbd2e699342791c7e4d80028`.

```text
image + language
    ↓
Eagle VLM (frozen)
    ↓
LayerNorm + VL self-attention transformer
    ├─ token sequence + state/action embeddings → DiT → action chunks
    └─ mean pool → embodiment-specific 4-layer MLP → tanh (64-d)
                      ├─ + state + action chunk → Q1, Q2
                      └─ + state                → V(s), critic training only
```

추가 transformer는 **원래 GR00T action head 안에 있는** `vlln`과
`vl_self_attention`입니다. critic을 위해 새로운 transformer를 처음부터 학습하는
구조가 아닙니다. BC에서는 이 모듈과 action projector, DiT를 학습합니다. critic
단계에서는 BC weight에서 VLM 및 LN/transformer를 복사하고 동결합니다.

코드 근거:

- [BC head](/home/yoon/vla_finetune/DEAS-Isaac-GR00T/gr00t/model/action_head/flow_matching_action_head.py):
  실제 sibling repo 경로는 `/home/yoon/vla_finetune/DEAS-Isaac-GR00T/` 아래입니다.
  `__init__`, `process_backbone_output`에 LN/transformer가 있습니다.
- `gr00t/model/gr00t_n1_deas_critic.py`, `from_pretrained`: BC의 `backbone`,
  `action_head.vlln`, `action_head.vl_self_attention`을 critic 모델에 복사합니다.
- `gr00t/model/action_head/deas_critic.py`, `set_trainable_parameters`,
  `process_backbone_output`, `forward`: 동결, pooling, 64-d projection 및 Q/V 학습입니다.
- `gr00t/model/action_head/deas_action_head_bon.py`, `get_action`: 한 번 추출한 특징으로
  DiT에서 여러 후보를 만들고 `min(Q1,Q2)` 기준 argmax 또는 temperature sampling합니다.

엄밀히는 BC actor와 critic을 **별도 모델/단계로 학습**한 뒤 추론에서 결합합니다.
공개 VLA 구현의 DiT 학습은 BC/filtered BC이며, Q gradient로 DiT를 함께 학습시키는
SVF actor 업데이트와 같지 않습니다. DEAS Q/V 출력은 scalar MSE가 아니라 101-bin
distributional loss입니다. 기본 Q는 512-width × 4-hidden-layer double MLP,
V는 256-width, depth-4 residual MLP입니다.

### N1.7 + fmrl/SVF에 적용한다면

N1.7에도 `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`의 `vlln` /
`vl_self_attention`이 있습니다. 현재 base config는 4-layer, 32-head, head-dim 64입니다.
따라서 N1.5 환경을 새로 만들 필요 없이 동일한 **특징 분기 방식**을 이식할 수 있습니다.

현재 `gr00t/rl/adapters.py`의 `FrozenGR00TEncoder`는 raw VLM token의 masked mean을
critic에 주므로 아직 위 DEAS 구조와 다릅니다. 이후 구현에서는 다음을 구분해야 합니다.

1. VLM은 동결하고 current/next observation당 한 번만 실행합니다.
2. critic에는 **BC 시점의 고정 LN/transformer**로 만든 특징을 주고,
   embodiment projection + Q/V를 학습합니다. SVF actor가 업데이트되더라도 critic의
   고정 특징 추출기가 의도치 않게 바뀌지 않아야 합니다.
3. SVF를 유지하면 outer Q(s,a), inner V(s,x,t)를 사용합니다. 이 time-conditioned
   inner V는 DEAS의 state-only V(s)와 다릅니다. 이것은 구조를 차용한 SVF이며
   DEAS 알고리즘 재현이라고 부르면 안 됩니다.
4. frozen critic-prefix를 포함한 checkpoint/resume와 최종 policy server adapter가
   필요합니다. 기존 SVF checkpoint를 표준 GR00T BC server에 그대로 넣을 수 없습니다.

## 데이터

모든 source revision은 `recipe.json`에 고정되어 있습니다. IPEC-COMMUNITY의
`*_no_noops_1.0.0_lerobot` LeRobot v2.1 공개 데이터를 사용합니다.

| Suite | Tasks | Episodes | Frames |
|---|---:|---:|---:|
| Spatial | 10 | 432 | 52,970 |
| Object | 10 | 454 | 66,984 |
| Goal | 10 | 428 | 52,042 |
| Long (`libero_10`) | 10 | 379 | 101,469 |

합계 1,693 episodes, source 약 1.89 GB입니다. 원본 raw 50 demos/task와 다르게
필터링된 공개 버전입니다. 20 Hz, image/wrist 256×256 RGB, state 8-d, action 7-d,
task instruction을 사용합니다. state는 EEF position/axis-angle/gripper이며 기존
modality key의 `roll/pitch/yaw` 이름만 보고 Euler angle로 재해석하지 않습니다.

Source는 RAID `datasets/libero_public/<suite>`에 보존합니다. 별도의
`datasets/libero_n17_bc/<variant>/<suite>`에 선택한 parquet/video만 symlink하고,
NVIDIA modality metadata와 **선택 subset만으로 재계산한** 통계를 저장합니다.
Goal episode 82 wrist video는 공식 patch를 overlay에만 연결합니다.

검증은 모든 selected episode의 state/action/길이/task 일치 및 파일 존재를 확인하고,
실제 GR00T CPU 로더로 task당 한 episode와 episode 82를 비디오 decode합니다.
전체 비디오를 전부 decode한 것은 아니며 `VALIDATION.json`에 검증 ID가 남습니다.
source는 수정하지 않고, 다른 provenance의 기존 경로는 덮어쓰지 않습니다.

```bash
cd /home/yoon/vla_finetune/Isaac-GR00T
# CPU only; full demos, all four suites. HTTP 429/5xx는 대기 후 제한적으로 재시도.
bash examples/LIBERO/finetune/run.sh data download-and-prepare

# 이미 다운로드한 원본에서 task당 1개 또는 5개를 재현 가능하게 선택.
bash examples/LIBERO/finetune/run.sh data prepare --demos-per-task 1 --seed 42
bash examples/LIBERO/finetune/run.sh data prepare --demos-per-task 5 --seed 42
```

`--suite libero_spatial`처럼 한 suite만 지정할 수 있습니다. `--seed`는 **데이터
선택 seed**입니다. suite별 BC 모델을 만들며, 네 suite를 섞는 unified-40-task
학습기는 아닙니다. HF credentials는 기존 CLI login을 사용합니다.

### Offline RL 데이터에 관한 주의

이 IPEC 변환본에는 reward/terminated/truncated가 없습니다. 따라서 BC READY와
RL READY를 구분하며, `READY.json`의 `rl_ready`는 false입니다. 마지막 frame을
검증 없이 성공/terminal로 꾸미지 않습니다.

추가 수집 없이 RL하려면 공개 reward-preserving 변환본 또는 원본 HDF5 라벨과
episode/frame alignment를 검증해서 붙이는 단계가 필요합니다. 원본 LIBERO
`scripts/create_dataset.py`는 마지막 transition에 reward=1, done=1을 기록하지만,
no-op 제거 후에도 같은 transition이 보존되는지는 별도로 확인해야 합니다.
reward 귀속(r_t vs r_{t+1}), termination/truncation, chunk 경계, 마지막 next observation을
명시해야 합니다. demo-only RL은 가능하지만 DEAS의 성공/실패 policy-rollout 혼합 데이터와
동일하지 않고, 실패 action에 대한 critic의 일반화도 자동으로 보장되지 않습니다.

## BC 실행 / 재개

GPU별 batch 32, 20,000 steps, LR 1e-4가 안전한 초기 설정이며 VRAM 한계 측정값은
아닙니다. vision/LLM 동결, projector/DiT/VL self-attention 학습입니다. LIBERO의 두
camera/token 수가 달라 carrot의 batch 256/GPU를 그대로 보장할 수 없습니다.
W&B project는 `libero-gr00t`, 기존 로그인 계정을 사용하고 carrot run ID는 재사용하지 않습니다.

```bash
# 기본은 명령 출력만: GPU/W&B/출력 디렉토리를 변경하지 않음.
bash examples/LIBERO/finetune/run.sh \
  --suite libero_spatial \
  --output /raid/yoon/vla_finetune/outputs/libero-spatial-bc

# carrot 작업 종료 후 할당된 GPU container 안에서 같은 명령에 --execute 추가.
# 별도 tmux에서 실행해야 SSH 종료와 독립적으로 유지됨.
```

실행 전 데이터 READY + VALIDATION을 확인하며, host 실행과 다른 GPU 학습 작업이
있는 경우 거부합니다. 경량 GPU 유지 프로세스는 기존 도구를 재사용합니다.
checkpoint는 **2,000 step마다 RAID에 모델을 남기고 최신 checkpoint의
Adam/scheduler/RNG만 유지**합니다. 8시간 budget에서 저장하고 종료하도록 하되,
컨테이너 강제 종료 직전 저장은 보장할 수 없습니다.

완전 저장된 checkpoint가 생긴 후 다음 container에서 재개하려면:

```bash
source /home/yoon/vla_finetune/activate_gr00t.sh
cd /home/yoon/vla_finetune/Isaac-GR00T
python examples/carrot_in_pot/resume_training.py \
  /raid/yoon/vla_finetune/outputs/libero-spatial-bc \
  --save-steps 2000 --max-run-seconds 28800
```

경로 이름과 달리 resume 도구는 checkpoint에 저장된 dataset/model/W&B 설정을
복원합니다. 같은 GPU 수로 재개해야 하며 다른 학습을 자동 종료하지 않습니다.
BC 평가 환경은 [local setup](../local/README.md)을 사용합니다. 이번 준비 과정에서는
추가 GPU 학습, rollout 수집, 성공률 측정을 실행하지 않았습니다.
