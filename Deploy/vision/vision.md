[환경 설치]
python -m pip install
  opencv-python==4.13.0.92 \
  pandas==3.0.2 \
  pyrealsense2==2.57.7.10387

↓

opencv       : 4.13.0
aruco exists : True
SUBPIX       : 1
pandas       : 3.0.2
pyrealsense2 : OK


================================================================================
VISION SYSTEM 정리
================================================================================


[1] CAMERA ↔ ROBOT BASE CALIBRATION
──────────────────────────────────────────────────────────────────────────────

목표
    Camera에서 측정한 3D 좌표
        ↓
    Robot Base 좌표로 변환할 수 있게
    T_BASE_CAMERA 구하기


방법
    Base 기준 위치를 알고 있는 ArUco marker 3개 이상 설치
        ↓
    각 marker center를 Camera 좌표계에서 측정
        ↓
    대응점 생성

        Camera points
        P_C = {p1_C, p2_C, p3_C, ...}

        Base points
        P_B = {p1_B, p2_B, p3_B, ...}

        ↓

    Kabsch / SVD rigid registration
        ↓

    R_BASE_CAMERA
    t_BASE_CAMERA
        ↓

    T_BASE_CAMERA


사용
    T_BASE_STICK
        =
    T_BASE_CAMERA
        @
    T_CAMERA_STICK


배운 점
    - 3D rigid transform은 최소 3개의 non-collinear 대응점이 필요함.
    - marker 개수가 많다고 무조건 좋아지는 것이 아니라,
      실제 marker center 위치를 정확히 알고 있는 것이 중요함.
    - 카메라 몸체 중심과 RGB optical center는 다름.
      Extrinsic은 "RGB optical frame" 기준으로 생각해야 함.
    - 카메라를 물리적으로 움직이지 않는 이상
      FPS 변경과 extrinsic calibration은 무관함.



================================================================================
[2] STICK ↔ ARUCO MARKER CALIBRATION
================================================================================

목표
    Marker pose만 검출해서
        ↓
    Stick 중심 pose를 얻기


Stick local frame
    origin = stick geometric center
    +Y     = tail → tip
    +Z     = 기준 면 바깥쪽
    +X     = right-handed


Stick1
    Marker ID0
    Marker ID1

Stick2
    Marker ID2
    Marker ID3


각 marker마다 필요한 것
    T_MARKER_STICK


즉
    T_CAMERA_STICK
        =
    T_CAMERA_MARKER
        @
    T_MARKER_STICK


Marker 2개를 동시에 보면서
    Marker0 ↔ Marker1 상대 pose 측정
        ↓
    실제 stick geometry와 결합
        ↓
    T_M0_S
    T_M1_S

같은 방식으로
    T_M2_S
    T_M3_S


배운 점
    - 같은 stick에 marker가 여러 개 붙어 있어도
      각 marker의 local frame orientation은 서로 다를 수 있음.
    - 단순 translation offset만 맞추면 안 되고
      rotation까지 T_MARKER_STICK에 정확히 들어가야 함.
    - 한번 물리적으로 확정된 marker→stick transform은
      FPS나 카메라 변경과 무관한 "geometry"임.



================================================================================
[3] SINGLE / DUAL STICK POSE
================================================================================

SINGLE
    Marker 하나만 보임
        ↓
    4 corners
        ↓
    IPPE PnP
        ↓
    pose candidate 여러 개
        ↓
    branch 선택
        ↓
    T_CAMERA_MARKER
        ↓
    T_CAMERA_STICK


DUAL
    Marker 두 개가 동시에 보임
        ↓
    4 corners + 4 corners
        ↓
    총 8 corners
        ↓
    하나의 Stick pose를 직접 PnP
        ↓
    T_CAMERA_STICK


신뢰도 개념
    DUAL
      > SINGLE


SINGLE branch 선택 시
    1. 이전 pose history가 있으면 history와 가까운 후보
    2. fresh start이면 workspace orientation prior 참고
    3. 그래도 없으면 reprojection error가 작은 후보


workspace_reference_final.csv
    ↓
    "정상 workspace에서 이 marker가 보였던 orientation들"
    저장


중요
    CSV는 좌표 보정값이 아님.

    주 역할
        fresh SINGLE
            ↓
        IPPE의 잘못된 branch 선택 방지


배운 점
    - planar marker PnP는 겉보기에는 비슷한 pose candidate가
      둘 이상 생길 수 있음.
    - reprojection error만 작다고 항상 실제 pose가 맞는 것은 아님.
    - temporal history / workspace prior 같은 추가 정보가
      branch ambiguity 해결에 효과적임.



================================================================================
[4] POSE QUALITY GATE / FILTERING
================================================================================

PnP 결과
    ↓
[REPROJECTION GATE]
    이미지와 너무 안 맞으면 reject
    ↓
[JUMP GATE]
    직전 pose 대비 너무 크게 튀면 reject
    ↓
[HISTORY]
    일시적 miss / reject 처리
    ↓
[FINAL SMOOTHING]
    Position EMA
    Rotation SLERP
    ↓
최종 Stick pose


현재 30 Hz 기준 예

SINGLE
    max position jump = 35 mm/frame
    max rotation jump = 17.5 deg/frame

DUAL
    max position jump = 10 mm/frame
    max rotation jump = 7.5 deg/frame


의미
    SINGLE은 불확실성이 크므로 gate를 널널하게
    DUAL은 신뢰도가 높으므로 gate를 더 강하게


Position smoothing
    p_filtered
      =
    alpha * p_new
      +
    (1-alpha) * p_previous


Rotation smoothing
    q_filtered
      =
    SLERP(q_previous, q_new, alpha)


현재 30 Hz
    POS_ALPHA = 0.2584
    ROT_ALPHA = 0.1938


중요하게 배운 점
    FPS가 바뀌면 "frame 기반 parameter"는 그대로 쓰면 안 됨.

    예)
        70 mm/frame @ 15 Hz
            ↓
        35 mm/frame @ 30 Hz

    둘 다 실제 의미는
        약 1.05 m/s


    miss reset도

        3 frames @ 15 Hz
            =
        6 frames @ 30 Hz

        둘 다 약 0.2 sec


    smoothing도 동일

        alpha = 0.45 @ 15 Hz
            ≠
        alpha = 0.45 @ 30 Hz

    FPS가 2배가 되면 같은 1초 동안 filter update도 2배이므로
    alpha를 다시 환산해야 같은 시간 응답을 유지할 수 있음.



================================================================================
[5] SINGLE ↔ DUAL HANDOFF CORRECTION
================================================================================

문제
    같은 실제 Stick인데

    SINGLE pose
        ≠
    DUAL pose

    약간의 systematic offset이 존재할 수 있음.


그냥 source를 바꾸면
    DUAL
      ↓
    SINGLE

순간에 pose가 몇 mm / 몇 deg 튈 수 있음.


해결
    DUAL이 신뢰 가능한 순간

    SINGLE pose ↔ DUAL pose 차이
        ↓
    correction을 천천히 online update
        ↓
    SINGLE 결과에 correction 적용


목적
    SINGLE ↔ DUAL 전환 시 pose continuity 확보


배운 점
    filtering은 단순히 센서 노이즈만 줄이는 게 아니라
    서로 다른 pose estimation source 사이의 handoff를
    부드럽게 만드는 데도 필요함.



================================================================================
[6] CAMERA BASE → HAND FRAME 변환
================================================================================

Vision에서 얻는 것
    T_BASE_STICK


정책이 원하는 것
    T_HAND_STICK


Indy FK
    joint angles
        ↓
    Forward Kinematics
        ↓
    T_BASE_HAND


그러면

    T_HAND_BASE
        =
    inverse(T_BASE_HAND)


최종

    T_HAND_STICK
        =
    T_HAND_BASE
        @
    T_BASE_STICK


Vision부터 한번에 쓰면

    T_HAND_STICK
        =
    inverse(T_BASE_HAND)
        @
    T_BASE_CAMERA
        @
    T_CAMERA_STICK


핵심 convention

    T_A_B
        =
    "B 좌표를 A 좌표로 변환"


배운 점
    좌표계 문제는 숫자를 맞추는 것보다
    transform의 방향을 일관되게 정의하는 것이 훨씬 중요함.

    T_BASE_CAMERA인지
    T_CAMERA_BASE인지

    항상 이름 자체에 방향을 명시하는 것이 안전함.



================================================================================
[7] QUATERNION / ORIENTATION 처리
================================================================================

Quaternion format
    [w, x, y, z]


중요
    q
    -q

    둘은 같은 rotation.


따라서 frame 간 quaternion을 그냥 숫자로 비교하면
    갑자기 sign이 뒤집힌 것처럼 보일 수 있음.


tracking에서는
    이전 quaternion과 dot product 확인
        ↓
    sign continuity 유지
        ↓
    SLERP smoothing


또한 square stick은
    local +Y 축 방향은 같아도
    Y축을 중심으로 90 deg roll 차이가 있을 수 있음.


배운 점
    "영상에서 평행해 보인다"
        ≠
    quaternion이 같아야 한다.

    물체 local frame 정의까지 같이 봐야 orientation을 해석할 수 있음.



================================================================================
[8] MAIN CAMERA + SIDE CAMERA 구조
================================================================================

MAIN camera
    = primary


SIDE camera
    = backup / fallback


Stick1
    MAIN에서 ID0 또는 ID1 하나라도 보이면
        ↓
    MAIN pose 사용

    MAIN에서 ID0, ID1 둘 다 안 보이면
        ↓
    SIDE pose 사용


Stick2도 동일
    ID2 / ID3


중요
    MAIN + SIDE pose를 평균내는 sensor fusion이 아님.


구조

            MAIN
              ↓
         Stick pose
              ↓
          primary
              │
              ├──────────→ FINAL

            SIDE
              ↓
         Stick pose
              ↓
          fallback
              ↑
        MAIN이 놓쳤을 때


두 camera가 동시에 볼 때는
    MAIN pose
    SIDE pose
        ↓
    dP / dR 비교
        ↓
    cross-validation 용도


배운 점
    두 카메라를 꼭 fusion해야 하는 것은 아님.

    occlusion 해결이 목적이면
        primary + fallback
    구조가 훨씬 단순하고 안정적일 수 있음.



================================================================================
[9] SOFTWARE SYNC
================================================================================

두 D435 RGB를 hardware sync하지 않고

각 capture thread에서
    frame 도착
        ↓
    time.monotonic_ns()
        ↓
    timestamp 저장


중요
    "processing 완료 시간"이 아니라
    "frame acquisition 시간"을 사용.


각 camera의 최근 result history에서

    MAIN timestamp
        ↕ nearest
    SIDE timestamp


가장 가까운 pair를 선택.


실제 결과
    nearest |dt|
        ≈ 5 ~ 8 ms


배운 점
    processing thread가 서로 다른 시간에 끝나더라도
    frame acquisition timestamp를 보존하면
    뒤에서 충분히 software synchronization 가능함.



================================================================================
[10] 30 Hz 만들면서 생긴 성능 문제
================================================================================

처음
    1280x720
    2 cameras
    CORNER_REFINE_APRILTAG
        ↓
    processed ≈ 4 Hz


APRILTAG refinement → SUBPIX 변경
        ↓
    sequential 2 cameras
        ≈ 18 Hz


MAIN camera만 처리
        ↓
    30 Hz


즉
    한 camera detector는 30 Hz 가능
    하지만

        MAIN processing
            ↓
        SIDE processing

    을 한 thread에서 순차 실행해서 병목 발생.


해결

    MAIN capture thread
          ↓
    MAIN vision thread


    SIDE capture thread
          ↓
    SIDE vision thread


    main thread
          ↓
    latest results / selector / GUI


결과
    각 camera vision processing
        약 25~28 ms

    30 Hz frame budget
        33.3 ms

    → 30 Hz 근처까지 처리 가능



================================================================================
[11] ArUco CORNER REFINEMENT 비교
================================================================================

CORNER_REFINE_NONE
    검출된 corner를 추가 refinement하지 않음

    장점
        가장 빠름

    단점
        corner 위치 정확도 낮을 수 있음


        ↓


CORNER_REFINE_SUBPIX
    검출된 corner 주변 pixel을 이용해
    sub-pixel 위치로 refinement

    장점
        빠름
        corner 정밀도 좋음

    현재 시스템
        검출 안정성 좋음
        pose 부드러움
        30 Hz 가능

    → 현재 선택


        ↓


CORNER_REFINE_CONTOUR
    marker contour 형태를 다시 이용해서
    corner 위치 refinement

    계산량 / 특성
        NONE과 SUBPIX보다 복잡
        contour 품질 영향을 받음


        ↓


CORNER_REFINE_APRILTAG
    AprilTag 방식의 corner refinement 사용

    장점
        어려운 조건에서 robust한 corner refinement를
        기대할 수 있음

    단점
        계산량 큼

    현재 실험
        두 camera에서 처리속도가 크게 감소
        ≈ 4 Hz

    → 현재 시스템에는 부적합


현재 선택
    CORNER_REFINE_SUBPIX


중요하게 배운 점
    "더 복잡한 refinement = 무조건 더 좋은 tracking"
    이 아님.

    실제 시스템에서는

        detection robustness
        pose jitter
        latency
        processing FPS

    를 같이 봐야 함.

    이번 환경에서는 SUBPIX가
        정확도 / 안정성 / 속도
    균형이 가장 좋았음.



================================================================================
[12] 현재 최종 VISION PIPELINE
================================================================================

                    D435 MAIN
                        │
                  capture @30 Hz
                        │
                acquisition timestamp
                        │
                 ArUco + SUBPIX
                        │
              SINGLE / DUAL tracking
                        │
                 filtering / smoothing
                        │
                 T_BASE_STICK_MAIN
                        │
                        │ primary
                        │
                        ├──────────────┐
                                       │
                                       ↓
                                  FINAL STICK
                                       ↑
                        ┌──────────────┘
                        │ fallback
                        │
                 T_BASE_STICK_SIDE
                        │
                 filtering / smoothing
                        │
              SINGLE / DUAL tracking
                        │
                 ArUco + SUBPIX
                        │
                acquisition timestamp
                        │
                  capture @30 Hz
                        │
                    D435 SIDE


FINAL STICK
    ↓
T_BASE_STICK
    ↓
Indy FK
    ↓
T_HAND_STICK
    ↓
RL Observation
    ↓
ONNX Policy @30 Hz



================================================================================
핵심적으로 얻은 것
================================================================================

1. Marker를 이용한 Camera ↔ Robot Base 3D extrinsic calibration

2. 여러 marker가 붙은 하나의 rigid object에서
   Marker ↔ Object transform을 명확히 정의하는 방법

3. planar marker PnP에는 pose ambiguity가 존재하며
   history / workspace prior가 branch 선택에 도움이 된다는 점

4. SINGLE / DUAL처럼 서로 다른 pose source 사이에서는
   handoff continuity가 중요하다는 점

5. Position은 EMA,
   Rotation은 quaternion + SLERP로 smoothing하는 것이 자연스럽다는 점

6. FPS가 바뀌면 jump threshold / miss count / smoothing alpha처럼
   frame-dependent parameter를 시간 기준으로 다시 생각해야 한다는 점

7. Multi-camera에서 capture timestamp를 유지하면
   processing을 병렬화해도 software sync가 가능하다는 점

8. Multi-camera는 항상 fusion할 필요가 없고,
   occlusion 대응 목적이면 MAIN + fallback 구조도 효과적이라는 점

9. 높은 정확도의 알고리즘이라도 latency가 너무 크면
   real-time control에서는 오히려 나쁜 선택일 수 있다는 점

10. 이번 시스템에서는
    AprilTag refinement보다 SUBPIX이
    정확도 / 안정성 / 30 Hz 처리 성능 측면에서 더 적합했음.
