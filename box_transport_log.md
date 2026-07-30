# box_transport_log.md — Indy-Wuji-Box-Transport

> Task별 큐레이션 로그. **원본(수정 금지)**: `nrmk_isaaclab_wuji/worklog.md`(2026-07-16, 07-22, 07-24).
> box↔chopstick은 운반·keypoint·orientation 보상 기계를 공유해 이 시기 기록이 겹친다 → chopsticks_grasp_log.md도 참조.

## 개요
- **task**: `Indy-Wuji-Box-Transport` — 파지한 상자를 world goal pose(위치+자세)로 운반.
- **상태**: 운반·keypoint·orientation 보상의 **개발 베드**. 여기서 검증한 기계를 chopstick으로 이식.
- **핵심 산출물**: best-so-far 위치 transport(φ), 결합형 orientation(goal 안에서만 지급, 드리프트 차단),
  SimToolReal **keypoint distance** 보상, 4-대칭 자세오차, 고정 palm-up goal, **quad wrap gate**(새끼 제외).

## 마일스톤 타임라인
- **07-16 (오후) 태스크 구현**: 사수님 방향으로 Box-Transport 신규. transport-φ(best-so-far 위치 포텐셜) 설계.
- **07-22 (7~8) keypoint 도입**: SimToolReal **keypoint 구현**(chopstick과 A/B). 최종 full-pose 8-corner
  + roll 4-sym(2점 방식 철회).
- **07-24 (2) euler/랜덤**: euler 축 분석, 랜덤 pose, box keypoint 진행, 분업 정리. box도 잔차 액션 전환.
- **(이후 세션) wrap+palm-up**: chopstick에서 시험한 **wrap 파지를 box로 이동**. box는 **quad wrap gate**
  (`balanced_quad_cage_gate`+`object_lift_in_balanced_quad_cage`, 새끼 제외) + **고정 palm-up goal**
  (roll 0·pitch 0·yaw 0.785398, resampling 1e9) + keypoint 거리.

## 핵심 교훈·함정
- **keypoint의 `d`는 정규화 안 된 미터** → φ(0~1)와 스케일이 100배+ 차이. weight 환산 시 예산(점)으로 볼 것.
- transport/orientation은 telescoping(best-so-far)이라 예산 = Δ×gate×W/30. gate형(1점)과 직접 비교 금지.
- **매니저는 리스트 안 SceneEntityCfg를 resolve 안 함**(`manager_base.py:398`) → quad gate는 named 파라미터로.
- 고정 goal은 `resampling_time_range=(1e9,1e9)` + ranges 축퇴로.

## 소스 포인터
- `nrmk_isaaclab_wuji/worklog.md`: 2026-07-16/22/24 (검색 "Box-Transport", "keypoint").
- 코드: `tasks/manipulation/grasp/box_mdp_cfg.py`, `box_transport_env_cfg.py`, `indy_wuji_box/env_cfg.py`.
