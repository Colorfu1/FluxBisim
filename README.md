# FluxBisim

`FluxBisim` is a dual-arm collaborative manipulation simulation platform built on `Isaac Sim 4.5.0`. Integrated with the `FluxVLA`, it provides a complete simulation workflow for embodied AI tasks, covering scene selection, skill composition, asset configuration, data collection, model training, deployment, and multidimensional evaluation.

https://github.com/user-attachments/assets/da04986a-4a37-473f-8d42-68ab41a3e47a

## 🌟 Features

- Centers on task composition, flexibly combining primitive skills such as pick-and-place, opening and closing, rotation, pushing and pulling, and handover to create complex dual-arm collaboration tasks that reflect real-world application needs.
- Uses atomic controllers and the simulation engine to automatically generate high-quality training data at scale.
- Integrates the `FluxVLA` to support training, deployment, and evaluation of mainstream models in high-fidelity simulation environments.

## Framework

<p align="center">
  <img src="./docs/framework.jpg" alt="Framework Architecture" width="800">
</p>

## 📢 Latest News

**\[2026/05/28\]** 🔥 FluxBisim has been open-sourced.

## 🛠️ Installation

### 1. Clone the Repository

```bash
cd FluxVLA
git clone https://github.com/FluxVLA/FluxBisim.git
```

### 2. Download 3D Assets

The 3D assets are hosted in the Hugging Face Dataset repository [`limxdynamics/FluxBisimAssets`](https://huggingface.co/datasets/limxdynamics/FluxBisimAssets). Download them to the `assets` directory in the project root:

```bash
cd FluxBisim
mkdir -p assets
hf download limxdynamics/FluxBisimAssets \
  --repo-type dataset \
  --local-dir assets
```

### 3. Install Isaac Sim

Download and extract Isaac Sim 4.5.0:

```bash
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip
unzip isaac-sim-standalone-4.5.0-linux-x86_64.zip -d /path/to/install_dir/
```

We recommend configuring an alias for the Isaac Sim Python environment:

```bash
echo 'alias benchmark_python="/path/to/install_dir/isaac-sim-4.5.0/python.sh"' >> ~/.bashrc
source ~/.bashrc
```

### 4. Install ROS Noetic

```bash
sudo apt update
sudo apt install ros-noetic-desktop-full ros-noetic-cv-bridge \
  ros-noetic-sensor-msgs ros-noetic-std-msgs python3-rosdep
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

After installing ROS for the first time, initialize `rosdep`:

```bash
sudo rosdep init
rosdep update
```

### 5. Install Python Dependencies

Install the dependencies into the Python environment bundled with Isaac Sim:

```bash
benchmark_python -m pip install --upgrade pip
benchmark_python -m pip install rootutils PyYAML h5py scipy
```

## Data Collection

```bash
cd /path/to/FluxVLA/FluxBisim
benchmark_python <collect_script> --env <env_name> --config <config_name> [--data_path <output_dir>]
```

Available tasks:

| Task                  | Script                                                        | Environment             | Config                                                                                                                                                                                                         |
| --------------------- | ------------------------------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pick Fruit to Plat    | `data_collect/pick_place_fruit/fruit_pick_place_collect.py`   | `kitchen`, `apartment`  | `apple_pick_place_config.yaml`, `banana_pick_place_config.yaml`, `carrot_pick_place_config.yaml`, `cucumber_pick_place_config.yaml`, `mangosteen_pick_place_config.yaml`, `whiteradish_pick_place_config.yaml` |
| Place Nut & Close Box | `data_collect/close_box/box_close_collect.py`                 | `industry`              | `box_close_config.yaml`                                                                                                                                                                                        |
| Store Apple in Drawer | `data_collect/pull_push_drawer/drawer_pull_push_collect.py`   | `apartment`             | `drawer_pull_push_config.yaml`                                                                                                                                                                                 |
| Screw Pitcher Lid     | `data_collect/screw_pitcher_lid/pitcher_lid_screw_collect.py` | `apartmentshort`        | `pitcher_lid_screw_config.yaml`                                                                                                                                                                                |
| Handover book         | `data_collect/handover_book/book_handover_collect.py`         | `apartment`, `industry` | `book_handover_config.yaml`                                                                                                                                                                                    |

Example:

```bash
benchmark_python data_collect/pick_place_fruit/fruit_pick_place_collect.py \
  --env kitchen \
  --config banana_pick_place_config.yaml
```

Data collected automatically by `FluxBisim` is saved in HDF5 format. To use it for `FluxVLA` training, convert it to the LeRobot Dataset v2.1 format by following the [data conversion guide](https://github.com/FluxVLA/FluxVLA/blob/main/docs/data_convert.md).

## 📊 Benchmark

### Leaderboard

| Codebase             | Pick Banana to Plate | Place Nut & Close Box | Store Apple in Drawer | Screw Pitcher Lid | Handover Book |
| -------------------- | :------------------: | :-------------------: | :-------------------: | :---------------: | :-----------: |
| FluxVLA (GR00T N1.5) |         0.96         |         0.88          |         0.88          |       0.33        |     0.45      |
| FluxVLA (Pi 0.5)     |         0.96         |         0.87          |         0.80          |       0.31        |     0.82      |

### Dataset

The training dataset is hosted in the Hugging Face Dataset repository [`limxdynamics/FluxBisimData`](https://huggingface.co/datasets/limxdynamics/FluxBisimData). It contains data automatically collected by `FluxBisim` and converted to the LeRobot Dataset v2.1 format, ready for `FluxVLA` training.

```bash
hf download limxdynamics/FluxBisimData \
  --repo-type dataset \
  --local-dir /path/to/FluxVLA/data/FluxBisimData
```

After downloading the dataset, set the data path in the `FluxVLA` training configuration to this directory.

### Model Evaluation

Run the simulation-based evaluation environment in `FluxBisim`:

```bash
cd /path/to/FluxVLA/FluxBisim
PYTHON_BIN=/path/to/install_dir/isaac-sim-4.5.0/python.sh ./fluxbisim_benchmark.sh pick_place_banana
```

Supported evaluation tasks are `close_box`, `handover_book`, `pick_place_banana`, `pull_push_drawer`, and `screw_pitcher_lid`.

In a second terminal, launch `FluxVLA` inference:

```bash
cd /path/to/FluxVLA
./scripts/fluxbisim_eval.sh \
  configs/pi05/fluxbisim/pi05_paligemma_pick_place_banana_full_finetune.py \
  /path/to/checkpoint.pth
```

The task ID is inferred from the configuration file name by `scripts/fluxbisim_eval.sh`, so make sure the configuration matches the evaluation task.

## 🙏 Acknowledgements

This project builds on the contributions of the following open-source projects and communities: [YCB Benchmark](https://www.ycbbenchmarks.com/), [Lightwheel SimReady](https://simready.com/), [X-Humanoid ArtVIP](https://huggingface.co/datasets/X-Humanoid/ArtVIP), and [Nvidia Omniverse USD](https://docs.omniverse.nvidia.com/usd/latest/index.html). If we missed your project or contribution, please open an issue or pull request so we can properly acknowledge it.

This project also acknowledges AgileX Robotics for providing the URDF model of the dual-arm robotic platform used in this work.

## Roadmap

- Release more evaluation metrics across three dimensions: spatial interaction, trajectory quality, and task outcomes.
- Support more dual-arm robots, such as Tron2.
- Strengthen domain randomization to improve data diversity and make model evaluation more comprehensive.
- Support data collection, model training, and evaluation for more scenes and tasks.
- Support data collection, model training, and evaluation for whole-body VLA models.
