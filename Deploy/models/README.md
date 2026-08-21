# `models/` — 배포용 ONNX 액터

학습 로그(`nrmk_isaaclab_wuji/logs/rsl_rl/<task>/<run>/exported/policy.onnx`)에서
**복사**해 둔 것입니다. 배포 코드가 로그 폴더 안을 직접 가리키면, 런을 정리하거나
이름을 바꾸는 순간 조용히 깨집니다 — 실제로 `run_policy` 의 기본 경로가 이미 사라진
런(`hand_final/2026-08-13_14-15-09`)을 가리키고 있었습니다.

파일명이 `<task>_<run>.onnx` 인 이유는 **어느 런에서 왔는지가 곧 이름**이어야 하기
때문입니다. `latest.onnx` 같은 이름은 두 달 뒤에 무엇인지 알 수 없습니다.

| 파일 | obs → action | 출처 |
|---|---|---|
| `hand_final_2026-08-21_01-14-36.onnx` | 105 → 20 | `logs/rsl_rl/hand_final/2026-08-21_01-14-36` |
| `hand_real_2026-08-21_10-32-51.onnx` | 105 → 20 | `logs/rsl_rl/hand_real/2026-08-21_10-32-51` |
| `finger_reach_2026-08-18_15-06-02.onnx` | 15 → 4 | `logs/rsl_rl/finger_reach/2026-08-18_15-06-02` |

새 정책을 넣을 때는 원본을 지우지 말고 복사만 하고, 이 표에 줄을 추가하세요.
학습 환경 파라미터가 필요하면 원래 런의 `params/env.yaml` 을 볼 것.
