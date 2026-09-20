# 이 서버에서 SimplerEnv + GR00T N1.7

N1.7 학습 환경은 변경하지 않고, 시뮬레이터/평가 클라이언트를 Python 3.10 환경에 분리한다.
GR00T에 고정된 **ManiSkill2 기반 SimplerEnv fork**를 사용한다. 별도의 ManiSkill3 GPU-parallel 버전이 아니다.

## 저장 위치와 범위

- 모델 서버: 기존 `~/raid/vla_finetune/envs/gr00t-n1.7` (Python 3.12, CUDA torch).
- 시뮬레이터/클라이언트: `~/raid/vla_finetune/envs/simpler-n17-client` (Python 3.10, SAPIEN 2.2.2).
- 클라이언트 torch는 CPU 빌드지만 **SAPIEN RGB 렌더링에는 GPU/Vulkan이 필요**하다.
- 모델 다운로드: `~/raid/vla_finetune/models/GR00T-N1.7-SimplerEnv-{Bridge,Fractal}`.
- 영상/로그: `~/raid/vla_finetune/outputs/simpler-n17/<실행시각>`.
- 설치 버전 기록: `~/raid/vla_finetune/logs/simpler-n17/installed-requirements.txt`.
- 이미 저장소에 있는 약 303 MB의 로봇/장면 자산을 재사용한다. 원본을 이동·수정하지 않는다.
- 대용량 Bridge/Fractal BC 데이터와 모델 weight는 설치 스크립트가 자동 다운로드하지 않는다.

설치 스크립트는 기존 환경 삭제, `uv sync`, `sudo`, 드라이버 설정 변경을 하지 않는다.
소스 리비전은 `manage.py`에서 확인하고, 전이 의존성까지 포함한 실제 설치 목록은 RAID에 기록한다.
요구사항 파일은 주요 호환 버전을 고정하지만 전체 transitive lockfile은 아니다.

## 1. 설치 및 CPU 점검 — 호스트에서도 가능

```bash
cd ~/vla_finetune/gr00t-bigenlight/n17
bash examples/SimplerEnv/local/setup.sh
bash examples/SimplerEnv/local/run.sh doctor
bash examples/SimplerEnv/local/run.sh tasks
```

`doctor`는 imports, GR00T RPC/rollout 코드, ffmpeg, 13개 등록을 확인한다.
성공해도 GPU reset/렌더링에 성공했다는 뜻은 아니다.

## 2. 컨테이너에서 렌더링 검증

```bash
nvidia-smi
bash examples/SimplerEnv/local/run.sh smoke --robot widowx
bash examples/SimplerEnv/local/run.sh smoke --robot google
# 두 기본 태스크가 통과하면 각 그룹 전체도 점검
bash examples/SimplerEnv/local/run.sh smoke --robot widowx --all-tasks
bash examples/SimplerEnv/local/run.sh smoke --robot google --all-tasks
```

`smoke`는 모델 없이 reset, RGB 크기/dtype, instruction, 3회 step을 확인한다.
성공률 측정이 아니라 렌더링/자산/로봇 제어 경로 점검이다.

**컨테이너 생성 단계에서** `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`가 필요하다.
`nvidia-smi`만 되는 compute-only 컨테이너는 Vulkan 렌더링이 안 될 수 있다.
이미 실행된 컨테이너 안에서 변수만 export해도 누락된 호스트 드라이버 라이브러리가 마운트되지는 않는다.
관리자/컨테이너 생성 설정에서 graphics capability와 Vulkan ICD/드라이버 노출을 확인해야 한다.
H200에서의 실제 렌더링 호환성·속도는 위 smoke 테스트로 검증해야 한다.
이 레시피는 호스트 GPU 접근 제한을 우회하지 않는다.

## 3. 공식 N1.7 baseline 모델 준비

```bash
bash examples/SimplerEnv/local/run.sh download --robot widowx
bash examples/SimplerEnv/local/run.sh download --robot google
```

각 모델의 커밋 SHA를 고정하고, 재개 가능한 Hugging Face 다운로드를 RAID에 저장한다.
원본 저장소에 함께 올라온 DeepSpeed/Adam/scheduler/RNG 파일은 제외한다.
Bridge 기준 safetensors는 약 6.91 GB이며 원본 전체 약 35.72 GB를 받을 필요는 없다.
RoboCasa/실물 carrot 체크포인트와 embodiment/정규화가 다르므로 그대로 바꿔 끼우지 않는다.

## 4. N1.7 평가 — 같은 GPU 컨테이너 안의 두 터미널

서버는 기존 N1.7 환경, 클라이언트는 새 simulator 환경을 자동 선택한다.
기본은 실행 없이 명령만 출력하며, 실제 실행할 때 `--execute`를 붙인다.

```bash
# 터미널 1: WidowX 모델 서버 (127.0.0.1:5555만 사용)
bash examples/SimplerEnv/local/run.sh server --robot widowx --execute

# 터미널 2: 기본 spoon_on_towel 10 episode, 1 env, seed 42
bash examples/SimplerEnv/local/run.sh eval --robot widowx --execute

# 다른 태스크 / 100 episode 평가
bash examples/SimplerEnv/local/run.sh eval --robot widowx \
  --task widowx_carrot_on_plate --episodes 100 --n-envs 5 --execute
```

Google은 서버와 클라이언트 양쪽 모두 `--robot google`로 실행한다.
두 로봇을 동시에 평가하려면 별도의 서버/GPU 자원과 서로 다른 `--port`를 지정해야 한다.
기본 action 실행 길이는 upstream 평가 예시대로 WidowX=4, Google=1, episode 제한=300이다.
`--action-steps`, `--max-episode-steps`, `--seed`, `--output-dir`로 변경 가능하다.
평가 로그의 성공률과 영상을 함께 점검한다. 이미 존재하는 출력 디렉터리는 덮어쓰지 않는다.
이 명령은 GR00T fork의 기본 visual-matching 평가이며 원 논문의 전체 variant sweep은 아니다.

## fmrl/SVF와의 관계

이번 준비 범위는 **시뮬레이터 + N1.7 BC 정책 평가**다. offline RL 학습까지 연결됐다는 뜻은 아니다.
후속 작업에는 (1) 성공/실패 rollout을 LeRobot 형식으로 기록하는 수집기,
(2) per-step reward·terminated·truncated·실제 next observation의 명시적 기록,
(3) SVF actor/critic을 로드하는 policy server adapter가 필요하다.
`gr00t/rl`의 SVF `.pt`를 위 `--model-path`에 넣어도 일반 BC 모델처럼 로드되지 않는다.
Bridge/Fractal BC 데이터에 terminal/reward가 있다고 가정하거나 성공 라벨을 만들어 넣지 않는다.
`info["success"]`와 termination은 서로 다르며 기존 GR00T wrapper의 구분을 유지한다.

참고: [GR00T의 평가 예시](../README.md),
[SimplerEnv upstream](https://github.com/simpler-env/SimplerEnv),
[NVIDIA 컨테이너 graphics capability](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html).
