# Neuromeka-IsaacLab
This repository is a Neuromeka IsaacLab extension for robot simulation, task configuration, and reinforcement-learning workflows. It provides:
- Neuromeka robot assets and IsaacLab task registrations
- Manipulation reach examples such as `Indy-Reach` and `Dual-Arm-Reach`
- Demo/deploy-style environments for Indy, Moby, NAMI, and Zen
- Shared environment, MDP, terrain, and utility modules for IsaacLab projects
- RSL-RL training and play scripts compatible with IsaacLab v2.3.0

NAMI navigation is one example in this repository, not the full scope of the project.

## Installation
### Prerequisite
#### 1. IsaacSim, IsaacLab
Follow the [IsaacLab Installation Guide](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/binaries_installation.html) for IsaacSim and IsaacLab installation.

The repository was tested with Isaac Sim **5.1** and IsaacLab **v2.3.0**.

Use the **prebuilt IsaacLab installation** with the **binary Isaac Sim installation**. This is the setup used for testing this repository. Do not use a source-built IsaacLab checkout unless you are intentionally developing against IsaacLab itself.

#### 2. Extra
After installing IsaacLab, a dedicated conda environment will be created (e.g., `env_isaaclab`).
Install extra packages.
```bash
conda activate env_isaaclab
pip install eclipse-zenoh open3d matplotlib pynput
```
Additionally, install [Git LFS](https://git-lfs.github.com/).

### Installing the neuromeka isaaclab extension
Clone the repository and install it as a package in the dedicated conda environment.
```bash
conda activate env_isaaclab
cd nrmk_isaaclab_public
git lfs pull
pip install -e .
```

## Usage Examples
### RSL-RL
Train an Indy reach policy.
```bash
python scripts/rsl_rl/train.py --task Indy-Reach --num_envs 128
```

Train a dual-arm reach policy.
```bash
python scripts/rsl_rl/train.py --task Dual-Arm-Reach --num_envs 128
```

Play a trained checkpoint.
```bash
python scripts/rsl_rl/play.py --task Indy-Reach --checkpoint /path/to/model.pt
```

View training logs.
```bash
tensorboard --logdir logs/rsl_rl
```

### Registered environments
Example task IDs registered by this package:
- `Indy-Reach`
- `Dual-Arm-Reach`

Simulation-only/demo task IDs for visualization, sensor streaming, and integration experiments:
- `Indy-Deploy`
- `Moby-Deploy`
- `Nami-Nav-Deploy`
- `Zen-Deploy`

### NAMI navigation example ([Demo video](https://youtu.be/EHRZnBG3YPo))
The NAMI navigation demo config is in `isaac_neuromeka/tasks/demo/nami_env_cfg.py`.

## Custom usecase
- For Indy reach tasks, start from `isaac_neuromeka/tasks/manipulation/reach/indy/env_cfg.py`.
- For dual-arm reach tasks, start from `isaac_neuromeka/tasks/manipulation/reach/dual_arm/env_cfg.py`.
- For shared manipulation logic, update `isaac_neuromeka/tasks/manipulation/common/env_cfg_common.py` and `isaac_neuromeka/tasks/manipulation/reach/reach_env_cfg.py`.
- For demo/deploy-style robot environments, check `isaac_neuromeka/tasks/demo`.
- For NAMI navigation scenes, update `NamiSceneCfg` in `isaac_neuromeka/tasks/demo/nami_env_cfg.py`. Currently, below three scenes are provided.
    - `isaac_neuromeka/assets/scene/hm3d_1`: [HM3D dataset](https://github.com/matterport/habitat-matterport-3dresearch)
    - `isaac_neuromeka/assets/scene/hm3d_2`: [HM3D dataset](https://github.com/matterport/habitat-matterport-3dresearch)
    - `isaac_neuromeka/assets/scene/nrmk_2nd_floor`: Neuromeka 2nd floor (scanned with [BLK2GO](https://shop.leica-geosystems.com/leica-blk/blk2go/overview?c1=GAW_SE_NW&source=USA_RC_BRND&kw=blk2go_exm&utm_source=google&utm_medium=cpc&utm_term=blk2go_exm&utm_campaign=USA__-__Reality_Capture__-__Branded&cr5=773061516399&cr7=c&gad_source=1&gad_campaignid=20547366468&gbraid=0AAAAADnuiFisxx-3b-ZPtg_ZwbxQbzfSl&gclid=Cj0KCQjwv-LOBhCdARIsAM5hdKePRWJ0ettjodu7fIahSKQtW6kiOjItG4iWOboSDgRcVMvLCbLRnZgaAgQuEALw_wcB))

Core files to look into are as follows:
- `isaac_neuromeka/tasks/demo/__init__.py`: Demo/deploy Gym environment definitions
- `isaac_neuromeka/tasks/demo/nami_env_cfg.py`: NAMI navigation environment
- `isaac_neuromeka/tasks/manipulation/reach/indy/__init__.py`: Indy reach Gym environment definition
- `isaac_neuromeka/tasks/manipulation/reach/indy/env_cfg.py`: Indy reach environment
- `isaac_neuromeka/tasks/manipulation/reach/dual_arm/__init__.py`: Dual-arm reach Gym environment definition
- `isaac_neuromeka/tasks/manipulation/reach/dual_arm/env_cfg.py`: Dual-arm reach environment
- `scripts/rsl_rl/train.py`: RSL-RL training script
- `scripts/rsl_rl/play.py`: RSL-RL play/export script

## Setting Up VSCode (Optional)

To configure VSCode for development, follow these steps:

1. Open VSCode and press `Ctrl+Shift+P`.
2. Select **Tasks: Run Task** and execute `setup_python_env` from the dropdown menu.
3. During execution, you will be prompted to enter the absolute path to your Isaac Sim installation.

This will generate a `.python.env` file in the `.vscode` directory, containing paths to all the Python modules provided by Isaac Sim and Omniverse. This enhances indexing and provides intelligent suggestions while coding.

*Note: Ensure your IsaacSim and IsaacLab paths are correctly configured in `tasks.json` or the prompt window.*

## Credits
This project includes modified code from [IsaacLabExtensionTemplate](https://github.com/isaac-sim/IsaacLabExtensionTemplate), which is licensed under the MIT License. See `LICENSE-MIT` for more details.

## Authors
- **Joonho Lee**, **Yunho Kim** at Neuromeka

## Acknowledgements
We sincerely appreciate the contributions of [Mayank Mittal](https://mayankm96.github.io/) for his support and insights in this project.
