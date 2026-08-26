# Isaac Lab Actuator PD Tuner

## 빠른 시작

- 용도: 관절별 `Kp`, `Kd`, `effort limit` 실시간 튜닝
- 입력: 주기적인 joint-position step command
- 출력: position, velocity, effort 그래프와 응답 지표
- UI: Isaac Sim 내부 UI가 아닌 별도 PySide6 GUI
- 저장: gain JSON, telemetry CSV

### 필요한 환경과 라이브러리

- 필수 환경: 호환되는 Isaac Sim 및 Isaac Lab
- 로봇 설정: `ArticulationCfg`가 정의된 Python 파일
- 로봇 리소스: 해당 asset이 사용하는 USD, URDF, mesh, Python package
- GUI 라이브러리: `PySide6`, `pyqtgraph`, `numpy`
- 버전 범위: `requirements-gui.txt` 참고
- 주의: Isaac Sim과 Isaac Lab은 이 패키지가 설치하지 않음

### GUI 라이브러리 설치

Isaac Lab 경로를 지정하는 경우:

```bash
cd pd_tuner
ISAACLAB_ROOT=/path/to/IsaacLab ./install_gui_dependencies.sh
```

현재 Python 환경에 Isaac Lab이 활성화된 경우:

```bash
cd pd_tuner
./install_gui_dependencies.sh
```

- 설치 대상 Python을 먼저 출력
- 사용자 동의 후 설치
- `sudo pip` 사용 안 함

### 실행

Isaac Lab 경로를 지정하는 경우:

```bash
ISAACLAB_ROOT=/path/to/IsaacLab ./run_pd_tuner.sh
```

Isaac Lab Python을 직접 지정하는 경우:

```bash
ISAACLAB_PYTHON=/path/to/python.sh ./run_pd_tuner.sh
```

### GUI 사용 순서

1. `Project root` 선택
2. 로봇 asset Python 파일 선택
3. `ArticulationCfg` 선택
4. Device 선택 후 `Start Simulation`
5. Joint 선택
6. Step amplitude와 period 설정
7. `Start step` 실행
8. Kp, Kd, effort limit 조정
9. `Save Tuned Gains`로 JSON 저장
10. `Export CSV`로 응답 데이터 저장

### 저장 위치

- 기본 출력 경로: `pd_tuner/outputs/`
- Gain: `outputs/gains/*.json`
- Session: `outputs/sessions/*.json`
- 원본 asset Python 파일: 수정하지 않음


----------------------------

This is a portable source package for tuning implicit-actuator position-drive
gains in a compatible Isaac Sim/Isaac Lab installation. Isaac Sim owns only the
simulation, articulation, rendering, and control loop. The tuning interface is
a separate Ubuntu desktop window implemented with PySide6 and pyqtgraph.

The tool does **not** create an `omni.ui` window or an Isaac Sim extension.

## What it measures

For one selected joint, the tuner holds every other actuated joint at a fixed
position target and alternates the selected target between `q0` and
`q0 + signed_amplitude`. It displays:

- target and actual joint position;
- actual joint velocity;
- Isaac Lab's implicit-PD `computed_effort` estimate;
- `applied_effort`, which is the estimate after actuator-limit clipping;
- the positive and negative effort limits;
- current/maximum error, peak velocity/effort, overshoot, 10–90% rise time,
  settling time, steady-state error, and saturation ratio.

Settling tolerance is `max(abs(step_amplitude) * 0.02, 0.001)`. If the current
half-period finishes without settling, the completed result remains
`Not settled`.

## Supported environment

Prerequisites:

- Ubuntu with a working Isaac Sim and Isaac Lab Python environment;
- a Python model definition exposing one or more `ArticulationCfg` objects;
- USD/mesh files and custom Python packages required by that asset;
- an implicit PD actuator with position targets and runtime drive writers;
- PySide6 and pyqtgraph for GUI mode.

The implementation was developed against:

- Python 3.11.15;
- Isaac Sim 5.1.0.0;
- Isaac Lab 0.54.3;
- PyTorch 2.7.0+cu128;
- NumPy 1.26.0.

Other compatible versions are allowed with a warning. Startup stops with a
clear error if these public articulation methods are absent:

```text
write_joint_stiffness_to_sim
write_joint_damping_to_sim
write_joint_effort_limit_to_sim
set_joint_position_target
write_data_to_sim
```

## Install GUI dependencies

Isaac Sim and Isaac Lab are not installed by this package. The optional GUI
packages are deliberately not installed without confirmation:

```bash
cd pd_tuner
./install_gui_dependencies.sh
```

The script identifies the same Isaac-enabled Python used by the launcher,
prints it, and asks before running pip. It never uses `sudo pip`.

Manual equivalent:

```bash
python -m pip install -r requirements-gui.txt
```

## Run

From the package directory:

```bash
./run_pd_tuner.sh
```

The launcher checks these choices in order:

1. `ISAACLAB_PYTHON`;
2. the active `python` environment;
3. `$ISAACLAB_ROOT/isaaclab.sh -p`;
4. `$ISAACSIM_PATH/python.sh`.

Examples:

```bash
ISAACLAB_ROOT=/path/to/IsaacLab ./run_pd_tuner.sh

ISAACLAB_PYTHON=/path/to/isaac-sim/python.sh \
  ./run_pd_tuner.sh
```

`PD_TUNER_PROJECT_ROOT=/path/to/project` may be used as the default GUI/CLI
project root while still allowing `--project-root` to override it.

The GUI can be pre-populated without changing tuner source:

```bash
./run_pd_tuner.sh \
  --project-root /path/to/robot_project \
  --asset-file /path/to/robot_project/robots/my_robot.py \
  --asset-cfg-name MY_ROBOT_CFG \
  --device cuda:0
```

An installed/editable package can also be launched with:

```bash
python -m pd_tuner
```

## GUI workflow

1. Select the optional **Project root**. It is added to the asset-import path
   only in Isaac child processes.
2. Select an asset directory or an individual Python file.
3. Wait for the isolated headless asset inspector to return the real
   `ArticulationCfg` objects. This avoids importing Omniverse modules in the Qt
   process.
4. Select a config and device, then press **Start Simulation**.
5. Select an actual spawned joint from `joint [actuator_group]`.
6. Set amplitude, period, delay, and direction; press **Start step**.
7. Change Kp, Kd, and effort limit. Updates are debounced to at most 20 Hz and
   applied at a physics-step boundary.
8. Inspect the three plots and current/completed transition metrics.
9. Save gain JSON and export the full-session CSV.

Selecting a new joint pauses the waveform, captures its current position as
`q0`, clears plot and metric history, displays its original gains, and restarts
from the low phase only when requested.

## Asset discovery and project roots

The first screen searches conventional directories below the current project:

```text
assets/
robots/
robot_assets/
source/*/assets/
*/assets/
```

No layout is required. The directory picker and individual file picker always
remain available. A selected module is dynamically loaded and its actual
`ArticulationCfg` objects are validated in an isolated Isaac process.

If an import fails, the error panel includes the missing module name. Select a
project root that makes the package importable, or install that package in the
same Isaac Lab environment. The tuner never copies or installs the user's
project.

## Joint and actuator resolution

Resolution happens after the articulation is spawned:

- `joint_names_expr` from every config actuator is recorded;
- the actuator object's resolved joint indices are mapped to actual spawned
  `robot.joint_names`;
- duplicate mappings, unmatched groups, and unactuated joints are reported;
- position/velocity limits and original per-joint Kp/Kd/effort limits are read
  from `ArticulationData`;
- only `ImplicitActuator` joints are currently tunable;
- unsupported actuator models remain visible as monitoring-only metadata.

The default Apply operation changes one selected joint. **Apply Gains to
Actuator Group** explicitly writes the same values to every joint in its group.

## Runtime application path

At the next physics boundary, the implicit-actuator adapter validates the
values and calls:

```text
Articulation.write_joint_stiffness_to_sim
Articulation.write_joint_damping_to_sim
Articulation.write_joint_effort_limit_to_sim
```

The adapter then updates the matching local implicit-actuator tensors. This is
necessary in the tested Isaac Lab version because the public writers update the
PhysX drive and `ArticulationData`, but deliberately do not update actuator
model tensors. Finally, the values are read back from `ArticulationData` and a
`gain_applied` acknowledgement is sent to the GUI. The GUI never labels its own
input as applied before that acknowledgement.

Validation requires finite values, `Kp >= 0`, `Kd >= 0`, `effort_limit > 0`,
and `step_period > physics_dt`. Large values are permitted for expert use but
should be increased cautiously.

## Effort plot semantics

For the supported implicit actuator:

```text
computed_effort
    Isaac Lab estimate: Kp*(q_target-q) + Kd*(qd_target-qd) + feed-forward

applied_effort
    computed_effort clipped by the actuator effort limit

measured_joint_effort
    unavailable in this API; not plotted and written empty/NaN in CSV
```

PhysX does not expose a measured joint motor torque through this implicit
actuator path. The tuner does not duplicate another signal under a false
`measured` label. `SATURATED` is shown at 98% of the active effort limit.

## Process and IPC architecture

```text
external PySide6 GUI parent
├── bounded control queue ───────────────► Isaac Sim child
├── bounded latest-first telemetry ◄─────┤ physics/control loop
├── bounded low-rate events ◄────────────┤ metadata/ack/errors
└── short-lived headless inspector child  (before simulation start)
```

All multiprocessing uses `spawn`. The simulation never performs a blocking
telemetry put; if the queue is full, an old sample is discarded. Physics runs
at the selected dt, telemetry defaults to 100 Hz, graph rendering to 25 Hz, and
gain edits are debounced to 20 Hz. GUI graph history is time-trimmed; CSV is
streamed to disk so memory does not grow with session duration.

Closing the GUI first requests a normal child stop. The child releases the
`SimulationContext` callbacks and singleton, then delegates timeline/stage
shutdown to `SimulationApp.close(skip_cleanup=True)`. Calling
`SimulationContext.stop()` separately is intentionally avoided: in the tested
Isaac Sim 5.1 build its `_timeline.stop()` can block in a spawned child. The
immediate Kit release also avoids the full global plugin-cleanup path, which can
wait indefinitely in that process topology. Parent-side process termination is
retained only as a timed crash fallback. A child crash and a telemetry timeout
are both surfaced in the status panel.

## Safety

The child pauses step input and physics on:

- non-finite selected-joint position, velocity, or available effort;
- configurable velocity-threshold violation;
- a hard position-limit violation;
- continuous effort saturation longer than the configured duration.

It does not automatically lower gains. The GUI remains available for reset,
restore, or shutdown. A floating-base articulation produces a warning because
base motion can invalidate a joint step response.

## Save, load, and outputs

The original asset Python file is never modified. Defaults are written below:

```text
outputs/
├── gains/
└── sessions/
```

Gain JSON records the absolute source asset path, config name, physics dt, and
per-joint gains. Use `--gain-config file.json` to apply one after spawn. Session
JSON stores GUI selections and can be loaded with:

```bash
./run_pd_tuner.sh --session examples/example_session.json
```

CSV columns are:

```text
simulation_time,joint_name,target_position,actual_position,joint_velocity,
position_error,computed_effort,applied_effort,measured_joint_effort,
stiffness,damping,effort_limit,saturated
```

## Headless logging

GUI packages are not needed in headless mode. A concrete asset, config, and
joint are required:

```bash
./run_pd_tuner.sh --headless \
  --project-root /path/to/project \
  --asset-file /path/to/project/robots/my_robot.py \
  --asset-cfg-name MY_ROBOT_CFG \
  --joint joint_name \
  --step-amplitude 0.2 \
  --step-period 2.0 \
  --duration 10
```

This runs the same spawned simulation child and saves CSV plus gain JSON.

## Known limitations

- This package does not contain or install Isaac Sim or Isaac Lab.
- The user's custom Python packages, USD files, and meshes must remain
  accessible.
- Runtime gain API changes across major Isaac Lab versions may require a new
  adapter in `pd_tuner/adapters/`.
- Only implicit position-drive actuators are tunable in this release. Explicit,
  learned, delayed, or remote actuator models are monitoring-only.
- A floating-base robot is not automatically welded or modified.
- The effort plot is an Isaac Lab estimate and clipped estimate, not a force
  sensor measurement.
- PyInstaller/single-binary packaging is intentionally out of scope because of
  Isaac Sim's runtime and extension loading. The deliverable is a portable
  source package for compatible Isaac Lab environments.

## Troubleshooting

**No compatible Python found**

Activate the Isaac Lab environment or set `ISAACLAB_ROOT`, `ISAACSIM_PATH`, or
`ISAACLAB_PYTHON`.

**Missing PySide6 or pyqtgraph**

Run `./install_gui_dependencies.sh`. It installs only after confirmation into
the selected Isaac Python.

**Asset import reports a missing module**

Choose the Python package's project root, or install the package in the active
Isaac environment. Do not point only to a nested asset directory when the
module uses absolute project imports.

**No tunable joints**

Inspect actuator warnings. The current adapter supports Isaac Lab
`ImplicitActuator` position drives only.

**The target clamps immediately**

Reduce amplitude or switch direction. Both requested and applied targets are
shown.

**The GUI reports telemetry timeout**

Check the Isaac child console for a spawn/API error. A physics safety pause can
also stop new samples until Resume/Reset.

## Remove completely

Delete the `pd_tuner` directory. The tool does not modify assets or write into
the selected external project. If GUI dependencies were installed solely for
this tool, uninstall them from the same Python environment manually:

```bash
python -m pip uninstall PySide6 pyqtgraph
```
