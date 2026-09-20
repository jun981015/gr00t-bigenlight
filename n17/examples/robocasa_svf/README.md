# N1.7 + fmrl/SVF: DEAS의 네 RoboCasa 태스크

선택한 방향은 **N1.5 DEAS 재현이 아니라 N1.7 + SVF**다. DEAS에서는 데이터 구성과
태스크 선택만 참고한다. 알고리즘은 `fmrl origin/dh`의 SVF 기본식을 이식한 `gr00t/rl`이다.
원본 fmrl 브랜치를 변경하거나 JAX 환경을 GR00T 환경에 섞지 않는다.

## 실험 구성

| 단계 | 데이터 | 모델 / 학습 |
|---|---|---|
| `bc24` | 24개 환경, expert demos 2,400개 | N1.7 초기 BC |
| `filtered-bc` | 아래 4개 환경의 demos + 성공 rollout | N1.7 추가 BC baseline |
| `svf` | 같은 4개 환경의 demos + 전체 rollout(성공/실패) | 추가 BC checkpoint에서 SVF 시작 |

태스크는 `CoffeeSetupMug`, `PnPMicrowaveToCounter`, `TurnOffStove`, `PnPCounterToMicrowave`.
공개 rollout은 **저자의 N1.5 정책이 수집한 데이터**다. 우리 N1.7이 수집한 rollout으로
간주하지 않는다. 네 태스크를 하나의 language-conditioned policy로 학습한다.

BC는 LLM/vision 고정, projector/DiT 학습. SVF는 frozen VLM 특징을 공유하고
actor/reference flow와 outer Q / time-conditioned inner V를 학습한다.
DEAS의 IQL/분포 critic이나 best-of-N 학습으로 바뀌는 것이 아니다.

## 환경과 데이터 위치

- 환경: 기존 `/raid/yoon/vla_finetune/envs/gr00t-n1.7` (Python 3.12 / torch 2.9).
- base model: `/raid/yoon/vla_finetune/models/GR00T-N1.7-3B`.
- 원본: `/raid/yoon/vla_finetune/datasets/{robocasa_mg_gr00t_100,deas_robocasa}`.
- N1.7 입력: `/raid/yoon/vla_finetune/datasets/robocasa_n17/`.
- 새 출력 예시: `/raid/yoon/vla_finetune/outputs/robocasa-n17-svf/`.
- 다운로드 로그: `/raid/yoon/vla_finetune/logs/robocasa-n17-svf/download.log`.

원본 BC 약 129.4 GB + offline 데이터 약 8.9 GB. revision SHA는 `recipe.json`에 고정.
N1.7 입력 디렉터리는 영상/parquet를 **symlink로 재사용**하고 별도 metadata/statistics만 만든다.
원본은 변경하지 않고, 누락 에피소드를 조용히 버리지 않는다.
세 카메라의 N1.7 별칭은 left→side_0, right→side_1, wrist→wrist_0으로 명시했다.
통계 계산은 low-dimensional 열만 읽어서 BC parquet에 포함된 대용량 이미지를 메모리에 모으지 않는다.

```bash
cd /home/yoon/vla_finetune/Isaac-GR00T
# GPU 불필요. 이미 받은 부분 재사용; HTTP 429는 서버 제한을 존중해 대기 후 재시도.
bash examples/robocasa_svf/download.sh
```

호스트 tmux `robocasa-n17-svf-download`에서 준비 중이면 중복 실행하지 않는다.
각 준비된 데이터에는 `READY.json`이 생긴다. 데이터 다운로드 완료와 GPU 학습 준비 완료는 다르다.

## 실행 (기본은 dry run)

`run.sh`는 기존 N1.7 환경을 활성화한다. **`--execute`를 추가하기 전에는 학습하지 않는다.**
실행 시 GPU 컨테이너 및 데이터 준비 여부를 확인한다. 로그는 각 output의 `train.log`.
실행 자체는 foreground이므로 컨테이너의 tmux 안에서 실행하는 것이 좋다.

```bash
# GPU 복구 후 첫 테스트: 실제 N1.7 BC 10 step, GPU당 batch 2
bash examples/robocasa_svf/run.sh bc24 --steps 10 --batch-size 2 \
  --output /raid/yoon/vla_finetune/outputs/robocasa-n17-svf/smoke-bc

# 초기 BC: 2 GPU × batch 16 = global 32, 30,000 step
bash examples/robocasa_svf/run.sh bc24 \
  --output /raid/yoon/vla_finetune/outputs/robocasa-n17-svf/bc24

# 초기 BC 완료 후 추가 BC: 동일 batch / step 기본값
bash examples/robocasa_svf/run.sh filtered-bc \
  --base-model /raid/yoon/vla_finetune/outputs/robocasa-n17-svf/bc24/checkpoint-30000 \
  --output /raid/yoon/vla_finetune/outputs/robocasa-n17-svf/filtered-bc

# SVF 10-step GPU smoke test: 학습된 RoboCasa actor checkpoint 필요
bash examples/robocasa_svf/run.sh svf --steps 10 \
  --base-model /raid/yoon/vla_finetune/outputs/robocasa-n17-svf/filtered-bc/checkpoint-30000 \
  --output /raid/yoon/vla_finetune/outputs/robocasa-n17-svf/smoke-svf

# 위 테스트가 통과한 후 --steps 30000 등으로 별도 output에서 본 학습
```

**SVF CLI는 현재 단일 GPU만 지원한다.** `torchrun`으로 두 개를 띄우면 DDP 학습이 아니므로
명시적으로 차단했다. 기본 batch 1은 VRAM 측정용 출발점이다. 두 번째 flow head와 K=8 SDE
rollout 때문에 이전 carrot BC의 GPU당 batch 256을 적용할 수 있다고 가정하지 않는다.
2-GPU SVF/DDP는 이번 준비에 포함되어 있지 않다.

SVF 기본: action chunk H=16, gamma=0.99, flow steps=10, K=8, kappa=1,
lambda multiplier=1, Q aggregation=mean, LR=3e-4, reference도 함께 학습.
RoboCasa에서 kappa/lambda를 튜닝한 결과는 아직 없으며 이는 시작 설정이다.
30,000 step도 로컬 초기 예산이지 fmrl 논문의 최종 benchmark protocol이 아니다.

## 보상과 종료 의미

- 실제 `next.reward` 0/1을 그대로 읽는다. N1.5 loader의 마지막 15 frame 성공 보상 확장이나
  reward-1 변환을 적용하지 않는다.
- 저자의 collection convention에 맞춰, 최종 `next.done` + 성공 보상은 terminated,
  성공 없이 끝난 rollout은 truncated로 구분한다. 임의 데이터에 쓰는 범용 규칙은 아니다.
- 공개 failed rollout에는 마지막 행동 이후 관측이 없다. 이 recipe는
  **`--no-bootstrap-on-truncation`을 명시**해 유한 episode 끝의 Q bootstrap을 0으로 둔다.
  실패를 성공으로 바꾸거나 reset 관측을 next observation으로 사용하지 않는다.
- `R = sum(gamma**i * r[t+i])`, `next_obs=o[t+16]`, `discount=gamma**16 * mask`.
  짧은 마지막 action chunk는 패딩으로 조작하지 않고 제외한다.
- `bootstrap_gamma`를 따로 설정하지 않는다. DEAS의 0.9/0.99 이중 discount와는 다른 SVF 설정이다.
- 모든 dataset은 같은 Panda Omron schema를 사용한다. SVF는 **BC checkpoint의 통계**를 유지하여
  actor와 reference/Q가 같은 normalized action 좌표계를 사용한다.

## 저장 / 재개 / W&B

W&B 프로젝트는 기본 개인 계정의 `robocasa-n17-svf`. 모델 artifact는 업로드하지 않고 RAID에 보관.

- 정규 checkpoint: 5,000 step마다.
- 복구용 저장: 첫 100 step, 30분마다, 최종 step/8시간 training-time budget 종료 시.
- BC: 이전 model checkpoint는 보존, 검증된 새 full checkpoint 이후 이전 Adam/LR/RNG만 정리.
- SVF: `model-step-N.pt`에는 Adam을 제외한 모듈 상태, `step-N.pt`에는 전체 learner/Adam/RNG.
  정규 milestone 모델은 남기고, `latest_resumable.json` 갱신 후 이전의 관리 대상 full 파일만 정리한다.
- SVF의 LR는 상수이므로 별도 scheduler는 없다. BC는 scheduler도 복구한다.
- 서버 강제 종료를 막는 기능은 아니다. 복구 저장 사이의 step은 유실될 수 있다.

BC는 output의 `latest_resumable.json`이 가리키는 checkpoint 디렉터리를 `--resume`에 전달한다.
SVF는 `checkpoints/latest_resumable.json`의 `file`을 `--resume`에 전달한다.
학습 목표 steps/batch/base model/데이터 의미를 확인하고 재개한다.

```bash
# 예시: 실제 포인터에서 확인한 경로로 바꿔서 실행
bash examples/robocasa_svf/run.sh svf \
  --base-model /raid/yoon/vla_finetune/outputs/robocasa-n17-svf/filtered-bc/checkpoint-30000 \
  --resume /raid/yoon/vla_finetune/outputs/robocasa-n17-svf/svf/checkpoints/step-5000.pt \
  --output /raid/yoon/vla_finetune/outputs/robocasa-n17-svf/svf-resumed
```

강제 종료 뒤 metrics가 checkpoint보다 앞서 있으면 **새 output에 재개**한다. 기존 log를 지우거나
동일 step을 겹쳐 기록하지 않는다. 새 output은 W&B run도 분리된다.
SVF checkpoint는 frozen VLM 본체를 포함하지 않으므로 원래 `--base-model`을 보존해야 한다.

## 검증 범위와 남은 작업

CPU에서 실제 CoffeeSetupMug 데이터 → N1.7 episode/transition loader → 작은 state-only SVF의
2-step 업데이트·저장을 확인했다. 이것은 **3B VLM/DiT를 GPU로 학습했다는 뜻이 아니다**.
테스트는 annotation, 다중 dataset 경계, sampler 재개, 실제 GR00T head adapter(작은 구성),
SVF 수식, checkpoint 복구/정리 등을 포함한다.

```bash
source /home/yoon/vla_finetune/activate_gr00t.sh
python -m pytest -q tests/gr00t/rl tests/gr00t/experiment/test_checkpoint_policy.py \
  tests/scripts/test_robocasa_svf_recipe.py --timeout=300
```

남은 검증: 전체 다운로드/준비 완료 → 2-GPU BC smoke → 학습된 actor로 단일 GPU SVF smoke/VRAM 측정.
RoboCasa 렌더링은 별도 simulator 환경에서 확인해야 한다. 기존 BC evaluation 경로는 사용할 수 있지만,
**SVF checkpoint를 평가 서버에 연결하는 masked-ODE policy adapter/export는 아직 별도 작업**이다.
현재 `step-N.pt`를 그대로 기존 BC 서버의 `--model-path`에 넘기면 안 된다.

자세한 알고리즘 계약은 [offline_rl.md](../../getting_started/offline_rl.md),
원본 비교 자료는 `fmrl`의 `origin/dh:docs/SVF_IMPLEMENTATION.md`를 참고한다.
