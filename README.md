# Chopsticks

Neuromeka Indy7과 Wuji Hand를 이용한 젓가락 조작 연구 코드.
Isaac Lab 기반 강화학습, MuJoCo 검증, 듀얼 D435 인식, 실물 정책 배포 구성.

## 시스템 구성

```text
+----------------------+      +----------------------+      +----------------------+
| Isaac Lab 학습       |      | 정책 내보내기        |      | 배포 실행             |
| RSL-RL PPO           +----->| checkpoint / ONNX    +----->| MuJoCo / Wuji Hand    |
| hand_real 계열 task  |      | 입출력 계약 고정     |      | 공통 PolicyRunner     |
+----------------------+      +----------------------+      +----------+-----------+
                                                                     |
                                   +---------------------------------+----------------+
                                   |                                                  |
                                   v                                                  v
                        +----------+-----------+                           +----------+-----------+
                        | 듀얼 D435 비전      |                           | 실물 손 backend     |
                        | Stick1/2 pose 추정  |                           | encoder·SDK·명령    |
                        +----------+-----------+                           +----------+-----------+
                                   |                                                  ^
                                   v                                                  |
                        +----------+-----------+      20D 잔차 액션         +----------+-----------+
                        | 105D 관측 조립      +-------------------------->| clip·scale·clamp   |
                        +----------------------+                           +----------------------+
```

## 주요 구성

| 경로 | 용도 |
|---|---|
| `nrmk_isaaclab_wuji/` | Isaac Lab 환경, task, reward, PPO 학습 코드 |
| `Deploy/` | MuJoCo·실물 정책 실행, ONNX adapter, 듀얼 카메라 인식 |
| `pd_auto_tuner/` | Wuji Hand 관절별 PD 자동 조정 |
| `pd_tuner/` | 수동 PD 조정 도구 |
| `활동기록/` | 날짜별 실험 조건, 결과, 판단 기록 |
| `wuji_test/` | Wuji Hand 기능·통신 확인 코드 |

## 학습·배포 흐름

```text
Task 설정
  -> Isaac Lab 병렬 학습
  -> best checkpoint 선정
  -> policy.onnx 내보내기
  -> read-only 관측·액션 확인
  -> MuJoCo 확인
  -> 짧은 실물 실행
  -> 장시간 실물 실행
```

## 정책 계약

- 실물 주력 actor 관측: `105D` 사용.
- 정책 출력: `20D` 관절 잔차 액션 사용.
- Stick pose: palm-frame `xyz+wxyz` 사용.
- Quaternion: canonical `wxyz` 사용.
- 정책 추론: `30 Hz` 사용.
- 실물 관절 명령: `90 Hz` 사용.
- Joint1/2 residual: `0.10 rad` 사용.
- Joint3 residual: `0.20 rad` 사용.
- Joint4 residual: `0.15 rad` 사용.
- 101D·103D checkpoint: 전용 adapter 확인 후 사용.

## 빠른 시작

학습 환경 이동.

```bash
cd ~/Chopsticks/nrmk_isaaclab_wuji
conda activate env_isaaclab
```

예시 학습 실행.

```bash
python scripts/rsl_rl/train.py \
  --task hand_real \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000
```

실물·MuJoCo 실행 명령은 [`Deploy/CLI.md`](Deploy/CLI.md) 사용.
학습·play·TensorBoard 명령은 [`CLI.md`](CLI.md) 사용.

## 모델 보관

- `policy.onnx`: 실물·MuJoCo 추론용 사용.
- `model_best.pt`: Isaac Sim play, 재학습, ONNX 재생성용 사용.
- `params/env.yaml`: 환경·보상·reset 조건 재현용 보관.
- `params/agent.yaml`: PPO 설정 재현용 보관.
- 최신 iteration보다 실제 성능이 좋은 best checkpoint 우선 보관.
- 전체 중간 checkpoint와 TensorBoard event는 Git 제외.

## 로그 관리

- `nrmk_isaaclab_wuji/logs/` 하위 task 폴더 구조만 Git 추적.
- 빈 디렉터리 유지용 `.gitkeep` 사용.
- checkpoint, event, CSV, 영상은 기본 제외.
- 공유가 필요한 best PT·ONNX만 별도 선별 추가.

## 안전 원칙

- 실물 최초 실행 시 `--read-only` 확인.
- 관절 순서, joint limit, action scale 일치 확인.
- 비전 STALE/LOST 발생 시 safe stop 사용.
- 전류·온도·관절 오차 확인 후 실행 시간 확대.
- observation, reward, reset, actuator 변경 시 checkpoint 호환성 재검토.

## 문서

- 배포 구조: [`Deploy/README.md`](Deploy/README.md)
- 배포 명령: [`Deploy/CLI.md`](Deploy/CLI.md)
- 전체 CLI: [`CLI.md`](CLI.md)
- Hand task 구조: [`nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/hand_grasp/README.md`](nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/hand_grasp/README.md)
- PD 자동 조정: [`pd_auto_tuner/README.md`](pd_auto_tuner/README.md)
- 날짜별 기록: [`활동기록/`](활동기록/)

## 주의

연구용 코드. 실물 장비 설정, 카메라 calibration, firmware, joint limit에 따라 결과가 달라짐.
배포 전 해당 checkpoint의 observation/action 계약과 학습 당시 `params` 확인.
