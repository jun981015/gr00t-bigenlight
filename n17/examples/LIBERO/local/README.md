# 이 서버의 LIBERO + GR00T N1.7

LIBERO 시뮬레이터/평가 클라이언트를 기존 N1.7 모델 환경과 분리한다.
실행 레시피는 **Spatial / Object / Goal / Long(`libero_10`) 각 10개, 총 40개 태스크**를 대상으로 한다.
원본 소스의 LIBERO-90도 설치·등록되지만 이 레시피의 suite 선택에는 포함하지 않는다.

2026-09-17 호스트 검증: 환경 설치(약 2.3 GB), 130개 등록/import, 4개 suite의 첫 태스크별
CPU reset/3-step 성공. 실행 설정 및 관련 CPU 회귀 테스트는 38 passed / 1 skipped.
로그는 `~/raid/vla_finetune/logs/libero-n17/cpu-smoke-<suite>.log`에 보관한다.
H200 EGL 카메라 렌더링과 실제 N1.7 정책 rollout은 아직 실행하지 않았다.
공식 모델 weight와 BC 데이터 다운로드도 아직 하지 않았다.

## 환경과 저장소

- 모델 서버: 기존 `~/raid/vla_finetune/envs/gr00t-n1.7` (CUDA PyTorch).
- 시뮬레이터: `~/raid/vla_finetune/envs/libero-n17-client` (Python 3.12, CPU PyTorch).
- Python 본체도 기존 RAID 런타임을 사용하여 호스트 `/usr/bin/python`에 의존하지 않는다.
- MuJoCo **3.3.1**, robosuite **1.4.0**, Gymnasium **0.29.1**, NumPy **1.26.4** 고정.
  MuJoCo를 최신 버전으로 무조건 올리면 robosuite의 `mj_fullM` 호출과 호환되지 않을 수 있다.
- 설정: `~/raid/vla_finetune/config/libero-n17/config.yaml` (`LIBERO_CONFIG_PATH`).
- 모델: `~/raid/vla_finetune/models/GR00T-N1.7-LIBERO/<suite>`.
- 평가 영상/로그: `~/raid/vla_finetune/outputs/libero-n17/<실행시각>/<task>`.
- 실제 설치 버전 목록: `~/raid/vla_finetune/logs/libero-n17/installed-requirements.txt`.
- 소스에 이미 있는 약 418 MB의 assets/init files는 그대로 재사용한다.

학습 환경, SimplerEnv 환경, `~/.libero`를 수정·삭제하지 않는다. `sudo`나 시스템 드라이버 변경도 하지 않는다.
주요 버전과 LIBERO/model 커밋을 고정한다. requirements는 전체 transitive lockfile은 아니므로 설치 목록도 함께 보관한다.
CPU PyTorch를 쓰는 것은 **클라이언트의 모델 의존성을 충족하기 위한 것**이며 GPU 이미지 렌더링을 CPU로 바꾼다는 뜻은 아니다.

## 1. 설치 / 호스트에서 가능한 점검

```bash
cd ~/vla_finetune/Isaac-GR00T
bash examples/LIBERO/local/setup.sh
bash examples/LIBERO/local/run.sh doctor
bash examples/LIBERO/local/run.sh tasks --suite libero_spatial

# GPU나 이미지 없이 실제 로봇 물리 시뮬레이션 reset/step
bash examples/LIBERO/local/run.sh cpu-smoke --suite libero_spatial
# 각 suite의 모든 태스크를 검사하려면 --all-tasks
```

`doctor`: 의존성/RPC/rollout import, ffmpeg, 전체 130개 태스크 등록을 확인한다.
`cpu-smoke`: 지정 태스크에서 로봇·물체 XML/mesh 로딩, reset, 3회 step, 유한한 state/reward를 확인한다.
**어느 것도 카메라 이미지나 H200 EGL 렌더링 검증을 대신하지 않는다.**

## 2. GPU 컨테이너에서 RGB 렌더링 점검

```bash
nvidia-smi
bash examples/LIBERO/local/run.sh smoke --suite libero_spatial
# 이후 나머지 suite도 같은 방식으로, 필요하면 --all-tasks
```

두 카메라(`video.image`, `video.wrist_image`)의 256×256 RGB, instruction,
reset/step을 확인하고 `~/raid/vla_finetune/logs/libero-n17/smoke`에 PNG를 저장한다.
이 테스트는 정책 없이 동작하며 성공률 평가가 아니다.

LIBERO는 MuJoCo **OpenGL/EGL** 렌더러를 쓰며 RTX ray tracing을 요구하지 않는다.
H200에서 실행하려면 컨테이너에 NVIDIA EGL/OpenGL 드라이버가 노출되어야 한다.
생성 시 `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`를 설정하고,
`libEGL_nvidia.so.0` 및 NVIDIA EGL vendor JSON을 확인한다.
생성 후 export만 해서는 누락된 호스트 드라이버 라이브러리가 추가되지 않는다.
`run.sh`는 `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`을 자동 설정한다.
장치 번호는 컨테이너의 EGL 열거 결과에 따라야 하므로 `MUJOCO_EGL_DEVICE_ID`를 임의로 고정하지 않는다.

## 3. N1.7 baseline 다운로드 — 자동 실행하지 않음

```bash
bash examples/LIBERO/local/run.sh download --suite libero_spatial
# 선택 가능: libero_object, libero_goal, libero_10
```

공식 `nvidia/GR00T-N1.7-LIBERO`의 고정 리비전에서 **선택한 suite만** 받는다.
suite당 weight 약 6.91 GB + 설정/정규화 통계이며 Adam/DeepSpeed checkpoint는 제외한다.
네 suite를 다 받으면 weight만 약 27.6 GB다. BC 학습용 대용량 데이터는 설치 과정에서 받지 않는다.

## 4. 평가 — 같은 컨테이너의 두 터미널

기본은 dry-run이며 `--execute`를 붙여야 실행한다. 기본 port **5556**으로 SimplerEnv의 5555와 분리한다.
서버는 loopback에만 bind하며 외부 포트를 열지 않는다.

```bash
# 터미널 1: 선택한 suite의 N1.7 모델 서버
bash examples/LIBERO/local/run.sh server --suite libero_spatial --execute

# 터미널 2: 첫 태스크 10 episode
bash examples/LIBERO/local/run.sh eval --suite libero_spatial --execute

# 같은 suite의 10개 태스크를 순서대로, 각각 50 episode
bash examples/LIBERO/local/run.sh eval --suite libero_spatial \
  --all-tasks --episodes 50 --n-envs 5 --execute
```

두 터미널의 `--suite`와 `--port`를 맞춰야 한다.
직접 fine-tune한 BC checkpoint는 서버의 `--model-path /절대/경로`로 지정한다.
단일 태스크 이름은 `tasks` 명령으로 확인한 뒤 `eval --task <이름>`으로 선택한다.
`--episodes`는 **태스크당** 횟수다. `--all-tasks`는 태스크들을 순차 평가한다.
출력에는 실행 설정/명령(`run.json`), 태스크별 `eval.log`, `exit_code`, 영상이 남는다.
기존 출력 디렉터리는 덮어쓰지 않으며 첫 실패에서 중단한다. GPU job/자동 재시작은 따로 만들지 않는다.

### 평가 프로토콜 주의

현재 GR00T 저장소 예시와 동일하게 action 실행 길이 **8**, episode 최대 **720**, seed **42**를 기본값으로 사용한다.
CLI에서 `--action-steps`, `--max-episode-steps`, `--seed`로 변경 가능하다.
현재 `LiberoEnv.reset()`은 환경 seed/reset을 사용하며 원본 `.pruned_init`의 고정 초기 상태를 순회하는 방식이 아니다.
따라서 논문의 고정 init-state 평가나 다른 VLA의 suite별 horizon 프로토콜과 동일하다고 주장하면 안 된다.
논문 수치 재현용으로 비교할 때는 이 초기 상태/step budget 차이를 먼저 맞춰야 한다.

## offline RL / fmrl-SVF

이번 범위는 환경과 **BC 모델 평가** 준비다. rollout을 offline RL용 LeRobot 데이터로 저장하거나
SVF actor/critic checkpoint를 policy server에 연결하는 기능은 별도 작업이다.
기존 `gr00t/rl`의 `.pt`를 일반 GR00T BC 서버의 `--model-path`로 바로 읽을 수 없다.
데이터를 모을 때 reward, 실제 next observation, success/terminated/truncated를 구분해서 기록해야 한다.

참고: [GR00T LIBERO 예시](../README.md),
[LIBERO upstream](https://github.com/Lifelong-Robot-Learning/LIBERO),
[MuJoCo 렌더링 문서](https://mujoco.readthedocs.io/en/3.3.1/programming/).
