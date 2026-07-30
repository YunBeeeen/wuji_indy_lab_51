# hand_grasp_log.md — hand_grasp / hand_setting

> Task별 큐레이션 로그. **원본(수정 금지)**: `nrmk_isaaclab_wuji/worklog.md`(2026-07-24~07-30),
> 루트 `ACTIVITY_2026-07-30.md`, `WORKLOG.md`. 상세는 원본 참조.

## 개요
- **task**: `hand_grasp` — 손만(hand-only)으로 2개 스틱을 파지·개폐(OPEN/CLOSE)하는 micro-skill.
  `pose_005`에서 리셋, 2개 dynamic rigid stick, 6 contact sensor, obs 103D, action 20D 잔차(scale 0.1), entropy 0.001.
- **상태**: **활성**. `12-21-21`을 OPEN/CLOSE **성공 기준선**으로 고정. `hand_setting`(Phase 2→3 전이) scaffold 신설.
- **위치(FSM 관점)**: 전체 8-phase 젓가락 조작의 **phase 2(파지 유지)+4(OPEN/CLOSE)** 검증용 micro-skill.

## 8-Phase FSM 계획 (2026-07-24 사용자 설계 — 이 task의 큰 그림)
```
0 stick acquire/orient → 1 palm-frame placement → 2 functional grasp →
3 object approach → 4 object pinch → 5 transport → 6 release → 7 terminal
```
- **완료 판정**: 순간 threshold 아닌 **dwell + hysteresis**. pinch는 접촉만이 아니라 "물체가 tip midpoint
  따라 움직이는 hold/slip"까지 요구.
- **obs**: phase one-hot 8D + elapsed/progress. 이전 phase 조건 = 다음 phase reward의 gate. phase 완료 보상 1회만.
- **학습**: 긴 단일 에피소드 X → phase별 저장된 유효 초기상태를 섞는 **phase-conditioned reset/curriculum**.
- 현재 `hand_grasp`=phase 2+4 검증. `hand_setting`=phase 2→3(thumb-seat→thumb-roll→support-close), obs 103→104D 예정.
- FSM 본 구현은 axial A/B 뒤로 미룸.

## 마일스톤 타임라인
- **07-24**: Phase 계획 정리(위 FSM).
- **07-27**: `hand_grasp` 시작. collision/valley probe, STATE B 잔차 RL.
- **07-28**: 6-contact 결합형 reward, pose_005 STATE B 유지 학습, 엄지 distal contact 교정, **OPEN/CLOSE mode-conditioned reward**.
- **07-29**: 연속 OPEN/CLOSE + **gap geometry 교정**(Stick2-axis gap), current-state 잔차 A/B, 각속도 페널티 +
  palm-relative 속도, distal tip **lateral gate**, hand_grasp_object scene scaffold.
- **07-30**: 아래 상세.

## 2026-07-30 세션 상세
1. **6-contact 붕괴 진단**(run 21-05-32): 수렴(~1500it, contact 5.969/6, reward 1227) 후 **policy 전체 붕괴**
   (~3100it contact 2.94/6, reward 145). 다른 reward 위한 3-contact 선택이 아니라 붕괴로 판정.
2. **contact 보존 reward/termination**: contact recovery(mean 10, min 40), gap/lateral에 `min_i(c_i)` 곱,
   contact-collapse termination(6/6 acquire latch, 유지 ≤4/6 10step 지속 시 종료).
3. **gap/lateral 분리**: 결합 지수 커널 → gap은 `contact_gate*exp(-|gap-t|/0.005)`, lateral은 독립 penalty
   `-5*(1-exp(-e_lat/0.005))`(mode·contact 무관).
4. **Stick2 anchor**: Stick2가 이동·회전(고정 병목) → 독립 pose deviation penalty(pos/ori, bounded [-10,0]) +
   OPEN/CLOSE·mode에 anchor gate `exp(-e_pos/..-e_ori/..)` 곱. success hard limit(pos≤5mm, ori≤10°).
   Stick2는 계속 dynamic(고정 X) — 정책이 palm·엄지 중간마디·약지로 실제 고정해야.
5. **`21-05-32` exact baseline 복원**(사용자 결정: 강화 누적 말고 baseline 재현 후 수동 중단). reward 10·termination 4.
   직후 단일변수: CLOSE `3mm→0mm`, success tolerance `±3mm→±0.5mm`.
6. **`12-21-21` 성공 기준선 확정**: OPEN/CLOSE 원하는 대로 나옴 → 통째 백업(`backups/..._12-21-21_success/`), resume 안 함.
7. **axial(y 장축) 보완 A/B**: 별도 penalty 아닌 기존 gap+lateral 지수 커널 확장 `exp(-e_gap/..-e_lat/..-e_axial/..)`.
   `e_axial`=distal-tip 차벡터의 Stick2 +y 성분 vs `pose_005` 기준(-4.8016mm) 절대오차. fresh 실행.
8. **`hand_setting` scaffold**: Phase 2→3 전용 task. `HandSettingEnvCfg`가 `HandGraspEnvCfg` 상속(동일 hand·스틱·
   센서·103D/20D). PPO만 분리(log ns `hand_setting`). 실제 setting MDP서 OPEN/CLOSE 2D → thumb-seat/roll/close 3D
   one-hot로 교체 예정(obs 103→104D). 그 전엔 scaffold checkpoint를 setting 결과로 해석 X.

## 핵심 교훈·함정
- **수렴 후 붕괴** 주의 — 6-contact 유지엔 recovery reward + collapse termination이 필요.
- **결합 지수 커널의 함정**: gap·lateral·contact를 다 곱하면 초기 shaping이 과결합 → 학습신호 상쇄. 분리할 것.
- **Stick2는 dynamic** — 정책이 실제 고정해야. reference rail 유지엔 pose deviation penalty + anchor gate.
- 강화 항 **무한 누적 금지** — baseline 재현(exact) 후 단일변수로 실험.
- 성공 기준선은 **통째 백업**(checkpoint/TB/policy/params/diff) 후 resume 안 하고 고정.

## 소스 포인터
- `nrmk_isaaclab_wuji/worklog.md`: 2026-07-24~07-30 (검색 "hand_grasp", "hand_setting", "Stick2", "OPEN/CLOSE").
- 루트 `ACTIVITY_2026-07-30.md`(오늘 활동), `WORKLOG.md`.
- 코드: `tasks/manipulation/hand_grasp/` (env_cfg, mdp). 백업: `backups/hand_grasp_*`, `..._12-21-21_success/`.
