# `models/` — 배포용 ONNX 액터

학습 로그(`nrmk_isaaclab_wuji/logs/rsl_rl/<task>/<run>/exported/policy.onnx`)에서
**복사**해 둔 것입니다. 배포 코드가 로그 폴더 안을 직접 가리키면, 런을 정리하거나
이름을 바꾸는 순간 조용히 깨집니다 — 실제로 `run_policy` 의 기본 경로가 이미 사라진
런(`hand_final/2026-08-13_14-15-09`)을 가리키고 있었습니다.

파일명이 `<task>_<run>.onnx` 인 이유는 **어느 런에서 왔는지가 곧 이름**이어야 하기
때문입니다. `latest.onnx` 같은 이름은 두 달 뒤에 무엇인지 알 수 없습니다.

| 파일 | obs → action | 출처 |
|---|---|---|
| `hand_real_2026-08-18_23-57-25_model4500.onnx` | 105 → 20 | `logs/rsl_rl/hand_real/2026-08-18_23-57-25` **model_4500.pt** — 가중치 대조로 확인(차이 0.000e+00). 실물 배포에 쓰는 것 |
| `hand_final_2026-08-21_01-14-36.onnx` | 105 → 20 | `logs/rsl_rl/hand_final/2026-08-21_01-14-36` |
| `hand_real_2026-08-21_10-32-51.onnx` | 105 → 20 | `logs/rsl_rl/hand_real/2026-08-21_10-32-51` |
| `finger_reach_2026-08-18_15-06-02.onnx` | 15 → 4 | `logs/rsl_rl/finger_reach/2026-08-18_15-06-02` |

2026-08-13의 `hand_final/2026-08-13_14-15-09(최종)/exported/policy.onnx`는
`101 → 20`인 옛 계약이다. 현재 105D 상수로 억지로 읽지 않고
`policy/legacy_hand_final_101.py` 전용 어댑터가 입력 폭으로 자동 선택된다.
`run_hand_policy_real`/`run_mujoco_policy`의 기존 `--policy` CLI를 그대로 쓰며,
별도 `--legacy` 플래그는 없다.

**어느 체크포인트에서 나왔는지는 추정하지 말고 대조할 것.** 내보내기 시각과 체크포인트
시각이 며칠 어긋나 있을 수 있다. `actor_state_dict` 의 `mlp.*.weight/bias` 를 ONNX
initializer 와 직접 비교하면 정확히 한 체크포인트에서만 차이가 0 이 된다
(`torch` + `onnx` 가 함께 있는 환경은 `env_isaaclab`).

새 정책을 넣을 때는 원본을 지우지 말고 복사만 하고, 이 표에 줄을 추가하세요.
학습 환경 파라미터가 필요하면 원래 런의 `params/env.yaml` 을 볼 것.
