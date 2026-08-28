# 활동 기록

Indy7·Wuji Hand 젓가락 조작 연구의 날짜별 핵심 기록.
실험 조건, 결과, 판단, 다음 작업 중심으로 정리.

## 전체 리뷰

- [중간 리뷰](MIDTERM_REVIEW_2026-07-20.md): 2026-07-08~07-31 연구 분해와 파지 획득 구조 정리.
- [최종 리뷰](FINAL_REVIEW_2026-08-28.md): 2026-07-08~08-28 학습·비전·배포 전체 요약.

## 연구 흐름

```text
Reach
  -> Cube·Box grasp/transport
    -> Chopsticks functional grasp
      -> hand_grasp OPEN/CLOSE
        -> hand_setting acquisition
          -> hand_move·hand_real robustness
            -> MuJoCo·Vision·Real deployment
```

## 2026년 7월

| 날짜 | 핵심 내용 |
|---|---|
| [07-08](ACTIVITY_2026-07-08.md) | Isaac 환경 통일과 Reach 기준선. |
| [07-09](ACTIVITY_2026-07-09.md) | ManagerBased RL 최소 관측·보상 정리. |
| [07-10](ACTIVITY_2026-07-10.md) | Cube-Grasp task와 병렬 물리 설정. |
| [07-11](ACTIVITY_2026-07-11.md) | 물체 중심 거리에서 SDF cage로 전환. |
| [07-12](ACTIVITY_2026-07-12.md) | 최저 꼭짓점 clearance와 파지 자세 교정. |
| [07-13](ACTIVITY_2026-07-13.md) | Action clip·진단과 reward farming 제거. |
| [07-14](ACTIVITY_2026-07-14.md) | Table/reset 기하와 물리 probe 정리. |
| [07-15](ACTIVITY_2026-07-15.md) | Terminal success와 transport reward 구축. |
| [07-16](ACTIVITY_2026-07-16.md) | Box-Transport와 크기 일반화. |
| [07-17](ACTIVITY_2026-07-17.md) | Cube success 98.2%와 lift layer 검증. |
| [07-18](ACTIVITY_2026-07-18.md) | Orientation·keypoint reward A/B. |
| [07-19](ACTIVITY_2026-07-19.md) | WRAP cage와 긴 물체 기하 병목. |
| [07-20](ACTIVITY_2026-07-20.md) | `ACQUIRE → TOOL_READY → USE` 구조 확정. |
| [07-21](ACTIVITY_2026-07-21.md) | Balanced tripod Box-Transport 이식. |
| [07-22](ACTIVITY_2026-07-22.md) | Chopstick 4-대칭과 coupled orientation. |
| [07-23](ACTIVITY_2026-07-23.md) | Middle semantic surface 추가. |
| [07-24](ACTIVITY_2026-07-24.md) | Keypoint pose와 fixed/random 목표 분리. |
| [07-25](ACTIVITY_2026-07-25.md) | Random pose 판독과 Phase-1 축소. |
| [07-26](ACTIVITY_2026-07-26.md) | Penta에서 quad contact gate로 변경. |
| [07-27](ACTIVITY_2026-07-27.md) | `hand_grasp` 신설과 CEM pregrasp seed. |
| [07-28](ACTIVITY_2026-07-28.md) | `pose_005`, 6-contact, OPEN/CLOSE 구성. |
| [07-29](ACTIVITY_2026-07-29.md) | Gap·lateral·axial 분리와 103D 계약. |
| [07-30](ACTIVITY_2026-07-30.md) | Contact collapse 분석과 `hand_setting` 신설. |
| [07-31](ACTIVITY_2026-07-31.md) | Pair reference·thumb pivot Stage gate. |

## 2026년 8월

| 날짜 | 핵심 내용 |
|---|---|
| [08-02](ACTIVITY_2026-08-02.md) | Flip reward와 cage gate 분리. |
| [08-03](ACTIVITY_2026-08-03.md) | Goal gate와 Joint4 authority A/B. |
| [08-04](ACTIVITY_2026-08-04.md) | Empty-hand reward hacking 제거. |
| [08-05](ACTIVITY_2026-08-05.md) | Hand-setting sigma 교정과 첫 획득 성공. |
| [08-06](ACTIVITY_2026-08-06.md) | `hand_object` 신설과 cube reward 구성. |
| [08-07](ACTIVITY_2026-08-07.md) | Inherited drop 수정과 최초 cube hold. |
| [08-08](ACTIVITY_2026-08-08.md) | Stick disturbance와 파지 복구 개선. |
| [08-09](ACTIVITY_2026-08-09.md) | Hand-setting Stage gate와 관절별 progress. |
| [08-10](ACTIVITY_2026-08-10.md) | `hand_real` 105D sim-to-real 관측 신설. |
| [08-11](ACTIVITY_2026-08-11.md) | Directed-axis 101D legacy A/B. |
| [08-13](ACTIVITY_2026-08-13.md) | Stick1 5 mm A/B와 model 900 보존. |
| [08-16](ACTIVITY_2026-08-16.md) | Quaternion history 105D 최종 계약. |
| [08-17](ACTIVITY_2026-08-17.md) | 외란 curriculum·MuJoCo·ArUco 구축. |
| [08-18](ACTIVITY_2026-08-18.md) | Asset 불일치 진단과 Finger-Reach 신설. |
| [08-19](ACTIVITY_2026-08-19.md) | 첫 실물 backend와 90 Hz scheduler. |
| [08-21](ACTIVITY_2026-08-21.md) | Deploy 재편과 105D read-only 완주. |
| [08-22](ACTIVITY_2026-08-22.md) | 실물 20초 실행과 action 일치 검증. |
| [08-23](ACTIVITY_2026-08-23.md) | Checkpoint contract verifier 구성. |
| [08-24](ACTIVITY_2026-08-24.md) | 실물 실행기 버그·전류·model 2400 검증. |
| [08-25](ACTIVITY_2026-08-25.md) | Box 69D raw obs와 hand-setting reward 재설계. |
| [08-26](ACTIVITY_2026-08-26.md) | Tip force와 contact-face reward 추가. |
| [08-27](ACTIVITY_2026-08-27.md) | Hand-real 지표 판독과 face-axis 수정. |
| [08-28](ACTIVITY_2026-08-28.md) | Hand-setting 추가 학습과 Manus 분석 시도. |

## 상세 기술 기록

- `box_transport_log.md`: Box 운반 세부 기록.
- `chopsticks_grasp_log.md`: 젓가락 파지 세부 기록.
- `cube_grasp_log.md`: Cube 파지 세부 기록.
- `hand_grasp_log.md`: Hand-grasp 세부 기록.
- `hand_object_log.md`: Hand-object 세부 기록.
- `hand_setting_log.md`: Hand-setting 세부 기록.
- `study.md`, `thesis.md`: 논문·수식·기술 근거 보존.

## 기록 원칙

- 날짜별 문서는 목표·변경·결과·판단·다음 순서 사용.
- Run 평가는 latest가 아닌 best checkpoint 기준 사용.
- Reward·observation·reset·actuator 변경 시 재학습 호환성 명시.
- 상세 수식과 긴 실험 근거는 기술 기록에 보존.
