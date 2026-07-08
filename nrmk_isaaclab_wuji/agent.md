# Agent Notes

- Main shared baseline: `/home/lsc/IsaacLab` with `env_isaaclab`, Isaac Sim 5.1, IsaacLab 2.3.x.
- New working extension: `/home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji`.
- Do not modify shared installs or older user assets directly; copy into the new working extension first.
- Neuromeka public is the IsaacLab/Indy7 extension base. Wuji hand assets came from existing Retargeting/GeoRT-style folders.
- Current target direction: Indy7 + Wuji hand for future chopstick manipulation RL.
- USD variants:
  - `indy7_wuji_right.usd`: mesh collision baseline.
  - `indy7_wuji_right_simplified.usd`: Indy arm simplified, Wuji hand mesh.
  - `indy7_wuji_right_all_simplified.usd`: Indy arm simplified, Wuji hand reduced Cube colliders.
- Recommended first RL asset: `indy7_wuji_right_simplified.usd`. Use `all_simplified` if hand mesh collision causes instability.
- Current `INDY7_WUJI_RIGHT_CFG` still points to `indy7_wuji_right.usd`; switch `usd_path` in `isaac_neuromeka/assets/indy.py` to test another variant.
