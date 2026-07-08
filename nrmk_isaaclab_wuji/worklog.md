# Worklog

## 2026-07-08

- Confirmed active baseline is existing `/home/lsc/IsaacLab` + `env_isaaclab` + Isaac Sim 5.1.
- Preserved shared `/home/lsc/IsaacLab` and `/home/lsc/isaacsim_pkg`.
- Created new working extension at `/home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji`.
- Copied Neuromeka public extension as Indy7/IsaacLab base.
- Added Wuji hand assets from existing Retargeting/GeoRT-style asset folders.
- Built combined URDF: `indy7_wuji_right.urdf`, fixed joint `tcp -> palm_link`.
- Generated and repaired USD assets so stage is `Z-up`, `metersPerUnit=1.0`.
- Added IsaacLab asset config `INDY7_WUJI_RIGHT_CFG` and task registration `Indy-Wuji-Reach`.
- Created three USD collision variants:
  - full mesh: `indy7_wuji_right.usd`
  - arm simplified + hand mesh: `indy7_wuji_right_simplified.usd`
  - arm simplified + reduced hand Cube colliders: `indy7_wuji_right_all_simplified.usd`
- Removed temporary/intermediate wrapper USD backups from the asset folder.
- Notes for chopstick RL: hand collision fidelity matters; start with arm-simplified + hand mesh, then compare all-simplified if mesh collision is unstable.
