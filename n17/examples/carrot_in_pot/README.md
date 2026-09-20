# carrot_in_pot → GR00T N1.7

UR7e + Robotiq 2F-85의 **RGB / joint-space BC 학습** 설정입니다.
학습 실행기는 다른 GPU 작업을 강제로 중단하지 않으며, 실제 학습은 수동으로 시작합니다.

## 데이터와 경로

원본은 [raw 기록](https://huggingface.co/datasets/Bigenlight/carrot_in_pot_raw)과
[LeRobot v3 변환본](https://huggingface.co/datasets/Bigenlight/carrot_in_pot_lerobot_v3)입니다.
v3 revision `a079393868f3f97915562309e9664f0941c37398`의 메타데이터와
state/action parquet, RGB 영상만 받았습니다(약 263 MB).
depth와 중복 raw 전체 파일은 받지 않았고, 토큰·환경 패키지는 변경하지 않았습니다.
54 episodes / 18,557 frames이며 train은 16,902 frames, val은 1,655 frames입니다.
변환본과 검증용 processor까지 약 295 MiB이며 모두 RAID에 있습니다.

```text
/raid/yoon/vla_finetune/datasets/
  carrot_in_pot_lerobot_v3/       # 내려받은 RGB subset 원본, 수정하지 않음
  carrot_in_pot_gr00t/
    train/                      # 49 episodes, 통계 fitting도 여기에서만
    val/                        # 5 episodes, seed=42, episode 단위 holdout
    source_mapping.json         # split/local episode ↔ source episode
    PREPARATION_COMPLETE.json
    VALIDATION.json             # 검증 성공 시에만 생성
    processor_preview/          # CPU 검증에 사용한 processor; 모델 checkpoint 아님
```

학습/검증은 서로 다른 디렉토리입니다. GR00T가 `info.json`의 split 이름만 보고
자동으로 분리해 준다고 가정하지 않습니다. 각 split의 episode/index를 연속 번호로
다시 매기되, `observation.state`, `action`, episode 내부 프레임 순서는 보존합니다.

## 입력 / 출력 의미

| GR00T 항목 | 원본 | 의미 |
|---|---|---|
| `video.scene` | `observation.images.cam1` | 외부 카메라 RGB |
| `video.wrist` | `observation.images.cam2` | 손목 카메라 RGB |
| `state.arm` | `ur_q1..ur_q6` | 측정된 follower 관절각, rad |
| `state.gripper` | `grip_pos` | 실제 gripper 상태, 0=open / 1=closed |
| `action.arm` | `cmd1..cmd6` | 절대 관절 목표값, rad |
| `action.gripper` | `grip_cmd` | 목표 gripper 명령, 0=open / 1=closed |

EEF delta teleop로 수집했어도 **저장된 action은 IK 이후의 관절 목표값**입니다.
EEF pose나 GELLO leader joint로 해석하면 안 됩니다. 디스크의 arm action은
절대값을 유지하고, 학습 processor에서 현재 측정 관절 상태 기준 상대값으로
변환/정규화합니다. gripper 명령은 절대값입니다. 배포 시 processor의 역변환이 필요합니다.

명령 문자열은 `Put carrot in pot`, action horizon은 **16 frames / 약 0.53초**입니다.
GR00T 내부에서는 `[40,132]`로 패딩되며 실제 유효 영역만 `[16,7]`입니다.

## 변환과 검증

다운로드는 이미 완료했습니다. 새 서버에서 같은 RGB subset을 재현하려면:

```bash
source ~/vla_finetune/activate_gr00t.sh
hf download Bigenlight/carrot_in_pot_lerobot_v3 \
  --repo-type dataset --revision a079393868f3f97915562309e9664f0941c37398 \
  --local-dir "$VLA_STORAGE_ROOT/datasets/carrot_in_pot_lerobot_v3" \
  --include 'README.md' 'meta/**' 'data/**' \
    'videos/observation.images.cam1/**' 'videos/observation.images.cam2/**'
```

변환 전용 명령은 다음과 같습니다(현재 서버에서는 이미 실행 완료).

```bash
source ~/vla_finetune/activate_gr00t.sh
cd ~/vla_finetune/Isaac-GR00T
python -m examples.carrot_in_pot.prepare_dataset
python -m examples.carrot_in_pot.validate_dataset
```

이미 변환된 디렉토리는 덮어쓰지 않습니다. 재변환하려면 `--output`으로 새 경로를
지정하고 검증에도 `--root`로 동일 경로를 지정합니다.

- v3의 이어 붙인 영상을 episode별 **H.264 CRF18, 1280×720, 30Hz CFR**로 재인코딩합니다.
  숫자 데이터는 bit-exact, 영상은 손실 압축이며 RGB 오차를 따로 측정합니다.
- 단순 stream-copy / timestamp passthrough는 경계나 마지막 프레임이 달라질 수 있어
  사용하지 않습니다. 영상마다 **실제 디코딩된 프레임 수**까지 확인합니다.
- 전 에피소드의 벡터 동일성, 각 카메라의 첫/중간/마지막 프레임 정렬, train 전용 통계,
  실제 GR00T processor/collator 배치를 CPU에서 검사합니다.
- 720p 두 카메라의 전체 시야를 유지하도록 학습 시 shortest edge=256,
  crop fraction=1.0을 사용합니다. 카메라가 16:9여도 강제로 정사각형으로 자르지 않습니다.
- 이 설정을 지원하도록 checkpoint processor 로딩 시 무시되던 전처리 override를
  수정하고 회귀 테스트를 추가했습니다. 기존 설정을 명시적으로 바꾸지 않으면 유지됩니다.

현재 변환본은 검증을 통과했습니다. 숫자 데이터는 전부 동일하고, 324개 RGB 샘플의
최대 평균 절대 오차는 0.941 미만(0~255 단위), 최소 상관계수는 0.9986입니다.
실제 processor/collator에서 state `[2,1,132]`, action `[2,40,132]`와 유효 action mask를
확인했습니다. 원본 전체 RGB를 픽셀 단위 무손실 보존한 것은 아닙니다.

`carrot_in_pot_gr00t_timestamp_debug/`는 초기 timestamp 문제를 조사하며 보존한
디버그 사본이며, **학습에 사용하지 않습니다**. v3 원본은 별도로 보존되어 있습니다.

## 실제 학습 시작

**23011 GPU 컨테이너 터미널에서**, 기존 fake 학습 종료 후 실행합니다.
이 스크립트는 두 GPU의 저부하 유지 프로세스를 자동 기동/재사용하고,
그 외 기존 GPU 작업이 있으면 실행기 자체가 종료됩니다. 다른 작업을 강제로 중단하지 않습니다.

### W&B 로그인과 기록

최초 한 번, 실행할 컨테이너 터미널에서 로그인합니다. API 키를 코드나 채팅에 넣지 마세요.

```bash
source ~/vla_finetune/activate_gr00t.sh
wandb login --verify
```

기본 W&B 프로젝트는 **`carrot-in-pot-gr00t`**이고 로그인한 계정의 기본 entity를
사용합니다. 다른 shell에서 `WANDB_ENTITY`가 설정되어 있으면 그 값이 우선하므로
개인 기본 계정을 쓰려면 `unset WANDB_ENTITY`를 실행하세요.
각 run 이름은 `carrot-bc-날짜시각-PID`로 고유하게 생성됩니다.
loss, learning rate, gradient norm과 W&B 시스템 지표를 기록하며, 이 launcher에
검증 루프는 없으므로 validation loss나 로봇 성공률은 자동으로 생성되지 않습니다.

- 로그: `~/raid/vla_finetune/outputs/carrot-bc-*/train.log`
- W&B 로컬 기록: 같은 run 디렉토리의 `wandb/`
- W&B 캐시 / 업로드 staging / 다운로드 artifacts: 모두 `~/raid/vla_finetune/` 아래
- `WANDB_LOG_MODEL=false`: 대용량 모델 checkpoint 자동 업로드 안 함
- `WANDB_WATCH=false`: 추가 파라미터/gradient histogram 수집 안 함

저장 경로 옵션은 [W&B 환경변수 문서](https://docs.wandb.ai/models/track/environment-variables)를
따릅니다. 원격 W&B 접속과 GPU 학습의 성공은 실제 run에서 별도로 확인해야 합니다.

### 실행

```bash
cd ~/vla_finetune/Isaac-GR00T
bash examples/carrot_in_pot/train.sh
```

기본값: **GPU 2개 / global batch 32 / 10,000 step / lr 1e-4**.
VLM의 vision tower와 LLM은 고정하고 action head 쪽을 학습합니다.
`torchrun` 2개 rank와 GR00T의 기존 ZeRO 설정을 사용합니다.

GPU당 **256개**로 별도 10,000step run을 시작하려면, 기존 GPU 학습이 정상 종료된 뒤:

```bash
bash examples/carrot_in_pot/train.sh --global-batch-size 512 --max-steps 10000
```

이는 사전학습 GR00T에서 시작하는 **새 학습**이며 기존 BC checkpoint의 재개가 아닙니다.
global batch는 두 GPU의 합이고 gradient accumulation은 1입니다. 기존 결과는 별도
디렉토리에 남고, 새 W&B run에 기록됩니다. LR 1e-4와 학습 모듈 및 저장 정책은 유지합니다.
새 run을 만들 때는 이전 실행의 `WANDB_RUN_ID`/`WANDB_RESUME` 환경변수를 재사용하지 마세요.

2026-09-15 기존 global batch 32 run의 10,000step 저장 및 정상 종료를 확인한 뒤,
`carrot-bc-20260915T125436-121903`을 global batch 512로 시작했습니다.
W&B run은 `junhyeong/carrot-in-pot-gr00t/jhyv5mmt`이며, 초기 38step 이상 정상 진행과
GPU당 약 80.4 GiB 사용을 확인했습니다(측정 시점 값이며 전체 학습 최대치는 아님).
기존 컨테이너 경과 시간을 고려해 이번 실행에는 `--max-run-seconds 28800`을 적용했습니다.

위 9월 15일 큰 배치 run은 4,294/10,000step에서 중단되었으며, 첫 저장
(당시 5,000step) 이전이어서 재개 가능한 checkpoint가 없습니다. 2026-09-17
사용자 승인으로 같은 사전학습 모델에서 큰 배치 학습을 처음부터 다시 시작했습니다:

- run: `carrot-bc-20260917T111806-264` (컨테이너 파일명 시각은 UTC)
- W&B: `junhyeong/carrot-in-pot-gr00t/slw7ufjk`
- GPU당 batch 256 / global batch 512 / 10,000step / 시간 예산 8시간
- 저장 간격 **2,000step**, 가장 최근 checkpoint만 전체 optimizer 상태 보관
- 컨테이너 tmux: `carrot-b512-20260917T111806`

이 새 큰 배치 run을 중단 후 이어갈 때는 다음 경로를 사용합니다:

```bash
bash examples/carrot_in_pot/train.sh --resume-dir \
  ~/raid/vla_finetune/outputs/carrot-bc-20260917T111806-264 --save-steps 2000
```

재개에는 첫 전체 checkpoint 저장이 필요하며, 배치 512 설정은 checkpoint에서 복원됩니다.

처음에는 `bash examples/carrot_in_pot/train.sh --max-steps 10 --global-batch-size 4`로
GPU smoke test를 할 수 있습니다. **2026-09-15 H200 2개에서 실데이터 10step 학습과
실제 DeepSpeed checkpoint 재개·시간 예산 종료 후 저장까지 검증했습니다.**

이번 실제 BC 학습은 fake 학습과 달리 checkpoint를 저장하도록 설정했습니다.
**2,000 step 간격 + 최종 저장**이고, **가장 최근 checkpoint 하나만 전체 학습 상태**를
보관합니다. 새 저장이 모든 rank에서 끝나고 파일 검증/완료 표시가 기록된 뒤에만,
이전 checkpoint의 Adam/DeepSpeed optimizer·스케줄러·RNG 상태를 삭제합니다.
이전 모델 가중치와 processor는 보존하며 정기 모델 checkpoint 개수 제한은 없습니다.
checkpoint와 processor는 `~/raid/vla_finetune/outputs/carrot-bc-*`에 저장됩니다.
저장 크기가 크므로 홈 디스크에 경로를 바꾸지 마세요.

### 컨테이너 시간 제한과 이어서 학습

기본 학습 시간 예산은 **학습 루프 시작부터 9시간**입니다. 예산에 도달하면 현재 step에서
정기 주기와 무관하게 전체 상태를 저장하고 정상 종료합니다. 저장 자체에는 추가 시간이 필요합니다.
서버의 10시간 제한이 **컨테이너 생성 시각 기준**이라면 이미 사용한 시간을 빼고,
남은 시간보다 충분히 짧게 `--max-run-seconds`를 설정해야 합니다. 이 타이머는 서버의
실제 할당 만료 시각을 조회하지 않습니다. 갑작스러운 SIGKILL/컨테이너 강제 종료 시에는
마지막 완료 checkpoint 이후의 진행분을 잃을 수 있습니다.

새 GPU 컨테이너에서 (동일한 RAID/home 마운트와 GPU 2개 필요):

```bash
cd ~/vla_finetune/Isaac-GR00T
bash examples/carrot_in_pot/train.sh --resume-dir \
  ~/raid/vla_finetune/outputs/carrot-bc-20260915T120408-84848
```

필요하면 뒤에 `--max-run-seconds 7200`처럼 이번 할당의 남은 시간에 맞는 예산을 지정합니다.
원래 checkpoint의 모델/데이터/배치/LR schedule/max_steps 설정을 불러오고, Adam·LR·RNG·
global step을 복원합니다. 동일한 W&B run으로 기록을 이어갑니다.
GR00T의 기존 sharded loader는 재개 시 `seed + global_step`으로 다시 섞으므로,
중단 없이 실행했을 때와 데이터 순서까지 bit-exact하게 같지는 않습니다.

- `latest_resumable.json`: 마지막 **완료된 전체 상태**의 포인터
- `checkpoint-N/resume_complete.json`: 파일 목록/크기, world size, W&B ID가 있는 완료 표시
- `checkpoint-N/model_only.json`: 이전 optimizer 상태가 정리되어 모델 용도로만 사용 가능
- `resume-*.log`: 재개 실행별 로컬 로그 (기존 `train.log`를 덮어쓰지 않음)

손으로 일찍 중단하려면 해당 run 디렉토리에 `STOP_AFTER_CHECKPOINT`라는 빈 파일을
만들면 됩니다. 다음 optimizer step 뒤 전체 저장 후 정상 종료하며, 다음 `--resume-dir`
실행 시 그 요청 파일은 이력을 남기고 소비됩니다. 이 기능은 새 저장 정책이 적용된 실행에만
사용할 수 있습니다. 재개 가능한 저장이 없거나 파일이 누락됐으면 처음부터 몰래 시작하지 않고 실패합니다.

2026-09-15 정책 변경 검증 과정에서 2,000/4,000step 모델도 보존했습니다.
시간 예산 저장 테스트로 4,001step checkpoint가 추가됐으며, 이후 정기 간격은 5,000step입니다.
실제 5,000step 저장 완료 후 최신 전체 상태가 `checkpoint-5000`으로 갱신되고,
4,001step의 optimizer 상태만 삭제되어 이전 모델 가중치는 남는 것까지 확인했습니다.

### GPU 유지 프로세스

학습 실행기는 컨테이너의 두 GPU에 유지 프로세스를 자동 기동하거나 기존 것을 재사용합니다.
현재 프로세스는 GPU당 약 534 MiB를 사용하며, 목표 연산 비율은 대기 중 10%,
다른 GPU 작업이 있을 때 1%입니다(커널 실행 시간/샘플링에 따라 실제 utilization은 다를 수 있음).
학습 종료 후에도 유지되며, 이를 완전히 끄려면 컨테이너 안에서 해당 PID를 확인하고 종료해야 합니다.
PID는 `~/vla_finetune/gpu_keepalive.pid`, `gpu_keepalive.gpu1.pid`, 로그는 RAID의
`logs/carrot-keepalive-*`에 있습니다. 재할당 후에는 PID가 달라지므로 예전 PID를 그대로 쓰지 마세요.
이 프로세스는 **10시간 할당 만료나 관리자 종료를 막지 못합니다**.

SSH 연결을 끊어도 유지하려면 컨테이너에서 tmux 세션 안에 실행하세요.
검증용 5개 에피소드는 학습에 포함하지 않았습니다. 위 launcher는 자동 검증 점수를
계산하지 않으므로, checkpoint별 open-loop 평가 또는 별도 로봇 평가가 필요합니다.
같은 세션/장면의 소량 데이터이므로 이 holdout만으로 일반화나 로봇 안전성을 판단하면 안 됩니다.

## Offline RL과의 관계

원본에 **reward / terminated / truncated 컬럼이 없습니다**.
모두 성공했다는 수집자 설명을 임의의 per-step reward나 terminal 라벨로 바꾸지 않았습니다.
이 설정은 우선 BC용이며, 이후 실제 라벨과 마지막 관측 처리 규칙을 제공하면
`gr00t.rl.dataset.LeRobotOfflineRLDataset`으로 연결할 수 있습니다.

학습된 BC checkpoint에 로봇 modality/정규화가 저장되므로, 앞서 준비한 SVF의
`--backend gr00t --model-path <BC checkpoint>` 시작점으로 사용할 수 있습니다.
depth, force/torque, TCP pose는 이번 모델 입력에 추가하지 않았습니다.
