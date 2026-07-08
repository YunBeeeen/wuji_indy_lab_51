# IsaacLab 5.1 작업 환경과 Indy7/Wuji 자산 메모

## 현재 기준 환경

- IsaacLab checkout: `/home/lsc/IsaacLab`
- Conda env: `env_isaaclab`
- Python: 3.11
- Isaac Sim pip packages: `isaacsim==5.1.0.0`, `isaacsim-rl==5.1.0.0`
- IsaacLab repo version: `2.3.2`
- 새 Indy/Wuji 작업 폴더: `/home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji`

기존 `/home/lsc/IsaacLab`, `/home/lsc/isaacsim_pkg`, hand/IL 관련 폴더는 수정하지 않는다.

## 폴더 역할 구분

`/home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji` 안에는 Isaac Sim을 설치하지 않는다. 이 폴더는 IsaacLab에서 불러 쓸 Neuromeka extension 복사본, Indy/Wuji URDF, task config, 공부 문서만 담는 작업공간이다.

실제 Isaac Sim은 다음 둘 중 하나를 사용한다.

- Python 실행/학습: `env_isaaclab` 안의 pip Isaac Sim
  - `/home/lsc/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim`
- GUI 실행/URDF import: 기존 standalone Isaac Sim 5.1
  - `/home/lsc/isaacsim_pkg/isaac-sim-standalone-5.1.0-linux-x86_64`

따라서 `wuji_indy_lab_51`에 `isaacsim` 폴더가 없는 것이 정상이다.

## 새 작업 폴더에 만든 것

- Neuromeka extension 복사본
  - 제외: `.git`, `logs`, `outputs`, `__pycache__`, `*.pyc`, `*.egg-info`
- Wuji hand 자산 복사본
  - `isaac_neuromeka/assets/model/urdf/wuji_right`
- 결합 URDF
  - `isaac_neuromeka/assets/model/urdf/indy7_wuji_right.urdf`
- IsaacLab asset config
  - `INDY7_WUJI_RIGHT_CFG`
- Gym task
  - `Indy-Wuji-Reach`

## Indy7 + Wuji 결합 기준

- Indy7 끝단 link: `tcp`
- Wuji hand root link: `palm_link`
- 연결 joint: `wuji_base_joint`
- 연결 방식: fixed joint, `parent=tcp`, `child=palm_link`
- 초기 attach pose: `xyz="0 0 0"`, `rpy="0 0 0"`

실제 손목/플랜지 정렬이 다르면 `indy7_wuji_right.urdf`의 `wuji_base_joint` origin만 조정한다.

## Collision 기준

Indy7 몸통/팔은 기존 `indy7_allegro_hand_right_simplified.usd`처럼 단순화된 collision 구성을 참고한다. Wuji hand는 visual mesh는 그대로 유지하고, 물리 접촉에는 `*_collision.STL` mesh를 사용한다.

학습에서 중요한 것은 visual mesh의 디테일보다 PhysX가 안정적으로 계산할 수 있는 collision이다. 손가락 접촉 학습을 할 때도 visual은 정확하게 두고, collision은 collision mesh나 convex/simplified mesh를 쓰는 편이 보통 더 안정적이다.

현재 `indy7_wuji_right.urdf`에는 Wuji `palm_link`와 모든 `finger*` link에 collision mesh가 추가되어 있다. 이 변경 뒤에는 Isaac Sim에서 URDF를 다시 import해서 USD를 갱신해야 collision이 USD에 반영된다.

## USD 만드는 방법

Isaac Sim GUI에서 URDF Importer로 아래 파일을 불러오면 된다.

```bash
/home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji/isaac_neuromeka/assets/model/urdf/indy7_wuji_right.urdf
```

권장 저장 위치:

```bash
/home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji/isaac_neuromeka/assets/model/usd/indy7_wuji_right/indy7_wuji_right.usd
```

`INDY7_WUJI_RIGHT_CFG`는 이 USD 경로를 바라보도록 추가되어 있다. USD 파일을 아직 만들지 않았다면 `Indy-Wuji-Reach` 실행은 실패하는 것이 정상이다.

## 실행 전 확인

새 extension을 env에 영구 설치하지 않고 먼저 `PYTHONPATH`로 테스트한다.

```bash
conda activate env_isaaclab
cd /home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji
PYTHONPATH=/home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji:$PYTHONPATH \
python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --num_envs 1 --max_iterations 1 --headless
```

기본 `python -c "import isaac_neuromeka"` 검증은 피한다. IsaacLab/Isaac Sim 계열 import는 `pxr`와 Kit path가 필요해서, 보통 `AppLauncher` 이후에 import하는 흐름이 안전하다.

## 삭제 후보

4.5/2.2.1 계열로 받은 설치물이라 삭제 가능한 후보:

- `/home/lsc/wuji_indy_rl_ws/isaacsim_4_5`
- `/home/lsc/chop_ws/isaac-sim`
- `/home/lsc/chop_ws/IsaacLab_2_2_1`

보존:

- `/home/lsc/IsaacLab`
- `/home/lsc/isaacsim_pkg`
- `/home/lsc/chop_ws/nrmk_isaaclab_migration`
- hand/IL 관련 기존 폴더

`/home/lsc/chop_ws/nrmk_isaaclab_public`는 새 작업 폴더에 복사되었으므로 나중에 삭제해도 되지만, 비교용 원본으로 당분간 남겨두는 편이 안전하다.
