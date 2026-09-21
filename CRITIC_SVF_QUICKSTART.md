# 새 critic + SVF 간단 안내

코드 브랜치: **`q-vgm-critic`**. 기존 actor/VLA의 입출력 규약은 유지하고,
action-conditioned IQL과 frozen-Q SVF 연결을 추가했다.
`qvgm10`은 요청한 Q-VGM-style 구조의 이름이며 논문 완전 재현을 의미하지 않는다.

## 기존과 달라진 부분

| 항목 | 기존 plain IQL | 이번 HF 모델 |
|---|---|---|
| 환경 Q | 첫 층 action concat, twin Q | 모든 층 action 주입, 독립 **10 heads** |
| Hidden | 512×4 + GELU/LayerNorm | **512→512→256**, GELU |
| Action | 패딩 포함 5280차원 | 유효 **16×7=112차원**만 선택 |
| IQL 학습 | EMA Q min → V, online V → Q | online Q mean → V, EMA V → Q |
| SVF 환경 Q | twin Q min | **10-head mean**, 고정 |
| Inner soft value | 2개 | 그대로 **2개** |

이번 weight는 **proprio 포함 IQL expectile 0.8 / 150k checkpoint**를 사용한다.
state는 pooled BC VLM 2048 + padded proprio 132 + embodiment 32 = **2212**.
학습/배포 모두 `NEW_EMBODIMENT`, 같은 BC processor와 정규화가 필요하다.
별도 `--exclude-proprio` 실험은 존재하지만 **아래 HF weight는 그 실험이 아니다.**

SVF 설정: 50ep/task BC, actor LoRA rank16/alpha32, batch32, flow4, candidates8,
κ=0.4, g=0.25(c=0.64), augmentation 없음. Flow/SDE noise는 유지한다.
Inner target은 후보 8개의 `lambda * logmeanexp(mean_10heads_Q / lambda)`.
Inner loss는 두 출력의 min, actor guidance는 두 inner 출력 평균의 action gradient다.

## HF 파일과 이름

- [SVF 5k](https://huggingface.co/RLobot-jun/gr00t-n17-bigenlight-50per-svf-qvgm10-iql-e08-q150k-proprio-k04-g025-noaug-step5000)
- [SVF 10k](https://huggingface.co/RLobot-jun/gr00t-n17-bigenlight-50per-svf-qvgm10-iql-e08-q150k-proprio-k04-g025-noaug-step10000)
- 공통 BC: `RLobot-jun/gr00t-n17-bigenlight-50per-task-step10000`
  revision **`1704897ac6a2a93c1d1fd806925b9237977ed8c5`**

이름의 `qvgm10-iql-e08-q150k-proprio`는 **환경 critic 구조·expectile·IQL 학습량·state 구성**,
`k04-g025-noaug`는 **SVF 설정**, 맨 뒤 `step5000/10000`은 **SVF 학습량**이다.
기존 DEAS/twin-Q HF 저장소는 바꾸지 않았다.

| 파일 | 내용 |
|---|---|
| `adapter/actor_lora.safetensors` | SVF actor LoRA, BC 본체 제외 |
| `adapter/env_q.safetensors` | 고정된 10-head 환경 Q (`q.` wrapper prefix) |
| `adapter/inner_critic.safetensors` | 학습된 2-head inner soft value |
| `adapter/adapter_config.json` | 정확한 구조·source checkpoint·BC revision |
| root `statistics.json`, `processor_config.json`, `embodiment_id.json` | 공통 BC 전처리 정보 |

Optimizer/target network는 제외된 **추론용**이다. 이 HF 저장소를 native BC model
directory처럼 그대로 서버에 넘기면 안 된다. 실물 관측에는 여전히 BC VLM forward가 필요하다.

## 코드 받기와 환경

```bash
git fetch origin
git switch q-vgm-critic
git pull --ff-only origin q-vgm-critic
source environments/activate_n17.sh  # 이 서버의 N1.7 환경
```

다른 컴퓨터에서는 N1.7 의존성을 설치하고 이 checkout의 `n17` package를 사용한다.
`activate_n17.sh`는 서버의 RAID 환경을 참조하므로 다른 컴퓨터에서 그대로 쓰지 않는다.

## 환경 Q weight 로딩 예시

다운로드한 HF 폴더를 `bundle_dir`로 지정한다. actor/inner weight 로더와 혼동하지 말 것.

```python
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from gr00t.rl.action_conditioned_iql import ActionConditionedQEnsemble
from gr00t.rl.frozen_q import FrozenActionIQLQ

bundle = Path(bundle_dir)
cfg = json.loads((bundle / "adapter/adapter_config.json").read_text())
spec = cfg["env_q_architecture"]
heads = ActionConditionedQEnsemble(
    spec["feature_dim"], spec["action_shape"], spec["action_indices"],
    spec["hidden_dims"], spec["num_q_heads"],
    exclude_proprio=spec["exclude_proprio"],
)
critic = FrozenActionIQLQ(heads)
critic.load_state_dict(load_file(str(bundle / "adapter/env_q.safetensors")), strict=True)
critic = critic.to(device).requires_grad_(False).eval()

# z_s: [B,2212], 같은 BC의 masked pooled conditioning + proprio + embodiment.
# action: [B,40,132], 같은 processor로 정규화한 clean action chunk.
action = action.detach().clone().to(device).requires_grad_(True)
q = critic({"features": z_s.to(device)}, action)[0]  # [B], ten-head mean
dq_da = torch.autograd.grad(q.sum(), action)[0]
```

State feature는 `FrozenGR00TStateExtractor` 또는 동일 BC 캐시에서 얻는다. LN/self-attention을
이미 적용한 캐시에는 다시 적용하지 않는다. Q는 FP32, action gradient는 **정규화 좌표 기준**이다.
위 코드는 Q scoring 예시이지 로봇 제어/actor rollout 코드는 아니다.

Actor는 공통 BC에 동일 LoRA 모듈을 만든 다음 `gr00t.rl.export_adapter.load_actor_adapter`
로 로드한다. **이미 SVF-finetune된 actor에 또 얹지 않는다.** Inner는 기존 `FeatureCritic`
2 heads / hidden512×4 / Fourier16 구조이고, env Q 클래스에 로드하지 않는다.

자세한 학습·validation·gradient 검증은
[ACTION_CONDITIONED_IQL.md](n17/examples/bigenlight_multitask/ACTION_CONDITIONED_IQL.md),
실물 배포의 남은 연결 작업은 [REAL_ROLLOUT.md](REAL_ROLLOUT.md)를 참고한다.
