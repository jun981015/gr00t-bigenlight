# RoboCasa DEAS baseline 준비

N1.5 DEAS baseline을 별도 환경에서 준비한다. 기존 `Isaac-GR00T` N1.7 환경과
`fmrl` 코드는 변경하지 않는다. 이 recipe가 N1.7/SVF 구현을 대신하지는 않는다.

## 데이터와 단계

1. `bc24`: NVIDIA N1.5 → 24개 환경의 expert demos 2,400개로 초기 BC.
2. `filtered-bc`: 초기 BC checkpoint → 아래 4개 환경의 demos + **성공 rollout만** 추가 BC.
3. `critic`: 추가 BC actor checkpoint의 특징을 고정 → demos + **성공/실패를 모두 포함한 rollout**으로 Q/V 학습.
4. `eval`: 같은 actor의 BC와 actor + critic의 best-of-10을 비교.

환경: `CoffeeSetupMug`, `PnPMicrowaveToCounter`, `TurnOffStove`, `PnPCounterToMicrowave`.
공개 rollout은 원 저자 정책이 수집한 데이터다. 우리가 학습한 초기 BC로 새로 수집한 데이터가 아니다.
공개 데이터 사용만으로 *동일 seed/동일 수집 정책까지* 재현했다고 주장하지 않는다.

HF 저장소와 commit SHA는 `recipe.json`에 고정했다.

- `kimtaey/robocasa_mg_gr00t_100`: 약 129.4 GB, 2,400 episodes. `total_tasks=288`은 언어 instruction 변형 수이며 환경 288개라는 뜻이 아니다.
- `changyeon/deas_robocasa`: 약 8.9 GB. `demos`, `success_rollouts`, `rollouts`를 구분한다.
- `nvidia/GR00T-N1.5-3B`: 약 5.4 GB.
- 원본 dataset class가 경로에 `robocasa`가 포함되면 성공 궤적 마지막 15 frame에 보상을 부여한다. 디렉터리 이름을 임의로 바꾸면 안 된다.
- reward/done의 horizon padding/terminal 처리와 loss는 upstream 그대로다. SVF와 비교할 때 이 의미를 별도로 맞춰야 한다.

## 저장 위치와 환경

기본 storage root: `/raid/yoon/vla_finetune` (`~/raid/vla_finetune`과 동일).
데이터는 `datasets/`, 모델은 `models/`, 새 가상환경은 `envs/deas-gr00t-n1.5/`.
출력 예시는 `outputs/deas-robocasa/`, 다운로드 로그는 `logs/deas-robocasa/download.log`.

```bash
cd /home/yoon/vla_finetune/DEAS-Isaac-GR00T
bash experiments/robocasa_deas/bootstrap.sh
source experiments/robocasa_deas/activate.sh
bash experiments/robocasa_deas/download.sh
python experiments/robocasa_deas/manage.py check --full
```

Python 3.10 / PyTorch 2.5.1 CUDA 12.4 / transformers 4.51.3를 사용하는 별도 환경이다.
설치된 정확한 패키지 목록은 환경 폴더의 `requirements-installed.txt`에 있다.
기존 N1.7 torch 2.9 환경에서 DEAS를 `pip install -e`하면 안 된다(둘 다 import 이름이 `gr00t`).
위 bootstrap은 upstream의 pinned 직접 의존성을 따르며, 간접 의존성 전체를 고정한 lockfile은 아니다.

GPU 컨테이너가 복구되면 FlashAttention 설치 및 CUDA 실행을 확인한다:

```bash
bash experiments/robocasa_deas/bootstrap.sh --flash-attn
source experiments/robocasa_deas/activate.sh
python -c 'import torch, flash_attn; print(torch.__version__, torch.cuda.device_count())'
```

## 실행 / 재개

### N1.5 영상 로더 최적화 (2026-09-20)

`video_backend=torchvision_av` 경로는 이제 keyframe seek로 필요한 구간만 디코딩하고,
선택된 frame만 RGB로 변환한다. PTS 기준 nearest-frame 선택(동률은 이전 frame),
순서, 중복 요청, episode 밖 timestamp의 경계 padding을 유지한다. seek가 실패하거나
필요한 이전 frame을 확보하지 못한 경우 처음부터 읽는 안전 경로로 돌아간다.

- 선택 frame LRU cache: **worker당 64 MiB** 기본값. worker 8개면 이미지 캐시 상한은
  합계 512 MiB이며, decoder·배치·prefetch 등 나머지 RAM은 별도다.
- `GR00T_PYAV_CACHE_BYTES=0`: 이미지 캐시 끄기. seek 최적화는 유지된다.
- `GR00T_PYAV_DECODE_MODE=full`: 기존 전체 영상 RGB 디코딩 경로로 되돌리기.
- 현재 episode parquet 캐시도 ID를 제대로 갱신해 연속 요청 때 재사용한다.
- 기존 DataLoader worker/prefetch와 랜덤 샘플링, augmentation, batch, LR, BF16은
  바꾸지 않았다. N1.7 shard sampler 자체를 이식한 것은 아니다.
- 기존 BC/critic/LIBERO/Bigenlight 실행 명령은 새 프로세스로 시작하면 자동 적용된다.
  이미 실행 중인 worker에는 소급 적용되지 않는다. 원본 영상과 checkpoint는 변경하지 않는다.

CPU 검증에서 16개 실제 filtered-BC 샘플의 모든 raw 값이 이전 로더와 bit-exact였고,
이미지 캐시를 끈 상태로 데이터 준비(full/seek + transform)가 8.71초 → 2.84초였다.
약 3.1배는 **CPU 준비 구간**의 개선이며 GPU 전체 학습 속도 보장이 아니다.
demo 영상의 긴 GOP 때문에 seek 후에도 여러 frame 디코딩이 필요한 경우는 남아 있다.

재현 (학습 시작/GPU 사용 없음):

```bash
source experiments/robocasa_deas/activate.sh
OMP_NUM_THREADS=1 python -m experiments.robocasa_deas.benchmark_video_loader
python -m pytest -q tests/test_pyav_frames.py tests/test_trajectory_cache.py
```

### 현재 실행: 24-task BC 생략, N1.5 base에서 바로 filtered BC

```bash
cd ~/vla_finetune/DEAS-Isaac-GR00T
bash experiments/robocasa_deas/run_filtered_bc_direct_1gpu.sh --execute
# 같은 학습 재개:
bash experiments/robocasa_deas/run_filtered_bc_direct_1gpu.sh --resume --execute
```

사용자 선택에 따라 첫 24-task BC를 생략한 실험이다. 초기 모델은
`models/GR00T-N1.5-3B`이며, 앞서 실행한 bc24 checkpoint를 사용하지 않는다.
4개 task의 demos 400개 + successful rollouts 182개(총 582 episodes)를 사용한다.
GPU 1개 / batch 32 / 30,000 step / action horizon 16,
W&B online / 로그 50 step / 정규 저장 5,000 step 설정이다.
기존 첫 100-step / 30분 / 8시간 복구 저장 정책도 적용된다.
출력: `/raid/yoon/vla_finetune/outputs/deas-n15-filtered-bc-direct-1gpu-b32`.

### H200 1 GPU에서 첫 24-task BC

```bash
cd ~/vla_finetune/DEAS-Isaac-GR00T
bash experiments/robocasa_deas/run_bc24_1gpu.sh --execute
# 컨테이너 재할당 후 최신 full checkpoint에서 재개:
bash experiments/robocasa_deas/run_bc24_1gpu.sh --resume --execute
```

이 런처는 GPU 1개, batch/global batch 32, 30,000 step, LR 1e-4,
cosine schedule / warmup 5%, action horizon 16, PyAV 로더를 사용한다.
W&B 기본 모드는 `online`, 프로젝트는 `gr00t-deas-robocasa`다.
학습 loss/LR 로그는 50 step마다 기록한다. 재개 시 저장/로그 간격 변경은 허용한다.
출력은 `/raid/yoon/vla_finetune/outputs/deas-n15-bc24-1gpu-b32`에 저장한다.
정규 저장 간격은 **10,000 step**이며 아래 첫 100 step / 30분 / 8시간
복구 정책도 적용된다. 새 full checkpoint 검증 후 이전 optimizer 상태만 정리한다.
첫 2-step 테스트는 `BC_RUN_STEPS=2 BC_RUN_OUTPUT=<새 경로>`로 분리한다.
`--execute`를 생략하면 명령만 출력한다. GPU 실행은 컨테이너에서 해야 한다.

24-task 공개 데이터의 task description에 `original_key`가 없는 경우,
로더가 LeRobot `task_index`를 통해 `meta/tasks.jsonl`의 문장을 읽는다.
원본 데이터의 parquet/video는 변경하지 않는다.

### 일반 단계별 런처

아래 명령은 **기본적으로 dry run**이다. 실행하려면 GPU 컨테이너에서 `--execute`를 추가한다.
각 단계는 별도 output을 써야 하며, 기존 비어 있지 않은 output을 덮어쓰지 않는다.

```bash
# 먼저 별도 output에서 10 step GPU smoke test (최종 full checkpoint도 저장)
python experiments/robocasa_deas/manage.py train bc24 \
  --output /raid/yoon/vla_finetune/outputs/deas-robocasa/smoke-bc24 \
  --max-steps 10 --batch-size 2

# 초기 BC
python experiments/robocasa_deas/manage.py train bc24 \
  --output /raid/yoon/vla_finetune/outputs/deas-robocasa/bc24

# 초기 BC가 30,000 step까지 완료된 뒤 추가 BC
python experiments/robocasa_deas/manage.py train filtered-bc \
  --base-model /raid/yoon/vla_finetune/outputs/deas-robocasa/bc24/checkpoint-30000 \
  --output /raid/yoon/vla_finetune/outputs/deas-robocasa/filtered-bc

# 추가 BC가 완료된 뒤 critic (먼저 별도 output으로 10 step smoke test 필요)
python experiments/robocasa_deas/manage.py train critic \
  --base-model /raid/yoon/vla_finetune/outputs/deas-robocasa/filtered-bc/checkpoint-30000 \
  --output /raid/yoon/vla_finetune/outputs/deas-robocasa/critic

# 예: 초기 BC 중단 후 재개. 원래 batch / max-steps / GPU 수를 그대로 사용.
python experiments/robocasa_deas/manage.py train bc24 \
  --output /raid/yoon/vla_finetune/outputs/deas-robocasa/bc24 --resume
```

기본값: GPU 2개, **GPU당 batch 16**, global batch 32, 각 단계 30,000 step.
LLM/vision은 고정, actor의 projector/DiT는 학습한다. 이전 carrot N1.7의 batch 256/GPU를
이 환경에 검증 없이 적용하지 않는다. critic은 현재/다음 RGB를 모두 처리하므로 actor와 메모리 사용도 다르다.
critic action horizon 16, discount1 0.9, discount2 0.99, expectile 0.7.
critic의 실제 Adam LR는 원본 실행 코드 기준 1e-4 (`critic_lr/value_lr` 선언값은 upstream에서 미사용).

W&B: 기존 로그인 자격을 사용하고 `gr00t-deas-robocasa` 프로젝트에 기록한다.
rank 0에서만 run을 열고 output의 `wandb_resume.json`에 run ID를 보관한다.
로그는 각 output의 `train.log`; 정상적인 프로세스 반환 시 `train.exit`에 종료 코드가 기록된다.
실행 명령은 foreground다. SSH 종료와 분리하려면 사용자가 컨테이너 안에서 tmux로 실행한다.

## 복구 체크포인트 정책

- 정규 저장: **5,000 step마다** standalone weights + 전체 학습 상태.
- 유실 방지 추가 저장: **첫 100 step**, **30분마다**, 단계 마지막 step.
- 실행 시작 후 8시간이 지나면 현재 step 종료 → full save → 정상 종료 (컨테이너 수명 보장은 아님).
- 모든 DDP rank의 RNG와 optimizer/scheduler가 쓰인 뒤에만 `latest_resumable.json`을 갱신한다.
- 새 full checkpoint가 검증되면 이전에 이 recipe가 저장한 checkpoint의 optimizer/scheduler/RNG만 정리한다.
  **이전 모델 가중치와 metadata는 유지한다.** 30분 recovery checkpoint의 모델도 유지하므로 RAID 사용량을 확인한다.
- 중단된 미완성 checkpoint보다 마지막 검증된 full checkpoint를 선택한다. 저장 중 컨테이너가 죽어도 이전 full 상태를 먼저 지우지 않는다.
- `STOP_AFTER_CHECKPOINT` 파일을 output에 만들면 다음 step 뒤 저장하고 멈춘다. 재개 전 파일을 다른 이름으로 옮긴다.
- 이미 학습이 끝난 output에서 max-steps를 바꿔 연장하는 기능은 별도 scheduler 정책 결정이 필요하므로 자동 지원하지 않는다.
- 강제 종료 직전의 미저장 step 복구나 서버 관리자에 의한 컨테이너 종료 방지는 보장하지 않는다.

## 원본 코드 대비 명시적인 수정

- 데이터와 단계별 output 경로 충돌 방지, dry-run / resume 검증 추가.
- 원본 CLI의 부모 프로세스·각 rank별 `wandb.init` 대신 `main()`을 직접 호출하고 HF rank-zero integration 사용.
- `TrainingArguments.report_to`가 list로 바뀌면서 W&B 설정이 TensorBoard로 덮이는 runner 분기를 수정.
- critic config가 자기 자신을 `critic_config`에 넣던 변수 shadowing 오류 수정.
- torch 2.5.1 / transformers 4.51.3 조합에서 RNG 재개가 막히는 오류 수정: restricted unpickling을 유지하고 NumPy RNG 타입만 한정 허용.
- 평가에서 `temperature` 인자가 policy까지 전달되도록 수정.
- **공유 특징 일치 보정**: critic을 최종 actor checkpoint에서 초기화하고, `eagle_linear`까지 포함한 backbone을 고정.
  평가가 actor backbone만 사용하는데 upstream critic은 다른 backbone에서 시작하고 projection을 업데이트할 수 있었기 때문이다.
- **정규화 일치 보정**: critic dataset의 state/action 정규화 통계를 actor checkpoint 통계에 맞춘다.
  평가 시 actor 통계로 정규화한 입력을 두 head에 공통 전달하기 때문이다.

마지막 두 항목은 upstream launch script와 다른 *명시적 일관성 보정*이다.
논문 숫자와의 완전 동일 재현이라고 부르지 말고 이 설정을 기록한다. loss 자체는 변경하지 않았다.

## 평가 준비 상태 / 남은 검증

훈련용 환경과 데이터 준비는 GPU 없이 가능하다. 하지만 **FlashAttention CUDA 실행, 2-GPU smoke test,
RoboCasa simulator 설치·asset 다운로드·EGL 렌더링은 아직 별도 검증이 필요**하다.
원본 README대로 simulator를 최신 버전으로 무조건 설치하면 과거 task/API와 달라질 수 있다.
특히 RoboCasa v0.2의 numpy==1.23.3/tianshou==0.4.10 요구는 이 GR00T 환경과 충돌한다.
따라서 `pip install -e robocasa`로 학습 환경을 자동 변경하지 않는다.
시뮬레이터 버전/의존성 조합은 GPU 컨테이너 복구 후 별도로 고정·검증한다.

평가 명령 생성은 준비되어 있다 (같은 네 task 각각 실행, 필요하면 seed 42/43/44로 반복):

```bash
python experiments/robocasa_deas/manage.py eval --task CoffeeSetupMug \
  --actor /raid/yoon/vla_finetune/outputs/deas-robocasa/filtered-bc/checkpoint-30000 \
  --critic /raid/yoon/vla_finetune/outputs/deas-robocasa/critic/checkpoint-30000 \
  --output /raid/yoon/vla_finetune/outputs/deas-robocasa/eval/CoffeeSetupMug-deas-s42
```

`--critic`을 빼면 BC 평가. seed 42/43/44는 로컬 평가 예시이며 원 논문의 정확한 seed라고 주장하지 않는다.
현재 기본은 각 task 50 episodes, 5 envs, 10 candidates, temperature 0.

CPU 테스트:

```bash
python -m pytest -q experiments/robocasa_deas/test_recipe.py
```

출처: [DEAS repository](https://github.com/csmile-1006/DEAS-Isaac-GR00T),
[paper](https://arxiv.org/abs/2510.07730),
[public offline rollouts](https://huggingface.co/datasets/changyeon/deas_robocasa),
[initial BC data](https://huggingface.co/datasets/kimtaey/robocasa_mg_gr00t_100).
