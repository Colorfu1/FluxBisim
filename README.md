# FluxBisim

`FluxBisim` 是基于 `Isaac Sim 4.5.0` 的机器人操作仿真环境，为 FluxVLA 提供仿真资产、数据采集脚本、控制器和 ROS 评测入口。

## 功能

- 基于 `Ubuntu 20.04`、`Isaac Sim 4.5.0` 和 `ROS Noetic`。
- 覆盖抓取放置、关箱、抽屉推拉、拧杯盖、书本传递等任务。
- 通过 ROS topic 连接 Isaac Sim 仿真环境与 FluxVLA 推理进程。

## 框架图

## 安装

### 1. 下载仓库

```bash
cd FluxVLA
git clone https://github.com/FluxVLA/FluxBisim.git
```

### 2. 下载 3D 资产

### 3. 安装 Isaac Sim

下载并解压 Isaac Sim 4.5.0：

```bash
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip
unzip isaac-sim-standalone-4.5.0-linux-x86_64.zip -d /path/to/install_dir/
```

建议为 Isaac Sim Python 配置别名：

```bash
echo 'alias benchmark_python="/path/to/install_dir/isaac-sim-4.5.0/python.sh"' >> ~/.bashrc
source ~/.bashrc
```

### 3. 安装 ROS Noetic

```bash
sudo apt update
sudo apt install ros-noetic-desktop-full ros-noetic-cv-bridge \
  ros-noetic-sensor-msgs ros-noetic-std-msgs python3-rosdep
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

首次安装 ROS 后初始化 `rosdep`：

```bash
sudo rosdep init
rosdep update
```

### 4. 安装 Python 依赖

依赖安装到 Isaac Sim 自带 Python 环境中：

```bash
benchmark_python -m pip install --upgrade pip
benchmark_python -m pip install rootutils PyYAML h5py scipy
```

## 数据采集

```bash
cd /path/to/FluxVLA/FluxBisim
benchmark_python <collect_script> --env <env_name> --config <config_name> [--data_path <output_dir>]
```

可用任务：

| 任务         | 脚本                                                          | 环境                    | 配置                                                                                                                                                                                                           |
| ------------ | ------------------------------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 水果抓取放置 | `data_collect/pick_place_fruit/fruit_pick_place_collect.py`   | `kitchen`, `apartment`  | `apple_pick_place_config.yaml`, `banana_pick_place_config.yaml`, `carrot_pick_place_config.yaml`, `cucumber_pick_place_config.yaml`, `mangosteen_pick_place_config.yaml`, `whiteradish_pick_place_config.yaml` |
| 关箱         | `data_collect/close_box/box_close_collect.py`                 | `industry`              | `box_close_config.yaml`                                                                                                                                                                                        |
| 抽屉推拉     | `data_collect/pull_push_drawer/drawer_pull_push_collect.py`   | `apartment`             | `drawer_pull_push_config.yaml`                                                                                                                                                                                 |
| 拧杯盖       | `data_collect/screw_pitcher_lid/pitcher_lid_screw_collect.py` | `apartmentshort`        | `pitcher_lid_screw_config.yaml`                                                                                                                                                                                |
| 书本传递     | `data_collect/handover_book/book_handover_collect.py`         | `apartment`, `industry` | `book_handover_config.yaml`                                                                                                                                                                                    |

示例：

```bash
benchmark_python data_collect/pick_place_fruit/fruit_pick_place_collect.py \
  --env kitchen \
  --config banana_pick_place_config.yaml
```

`FluxBisim` 自动采集的数据以 HDF5 格式保存。需要将其转换为 LeRobot Dataset v2.1 格式用于 `FluxVLA` 训练。基于[数据转换指南](https://github.com/FluxVLA/FluxVLA/blob/main/docs/data_convert.md)实现数据格式转换。

## 评测基准

### 排行榜

| Codebase             | Pick Banana to Plate | Place Nut & Close Box | Store Apple in Drawer | Screw Pitcher Lid | Handover Book |
| -------------------- | :------------------: | :-------------------: | :-------------------: | :---------------: | :-----------: |
| FluxVLA (GR00T N1.5) |         0.96         |         0.88          |         0.88          |       0.33        |     0.45      |
| FluxVLA (Pi 0.5)     |         0.96         |         0.87          |         0.80          |       0.31        |     0.82      |

### 数据集与模型权重

### 模型评测

启动 Isaac Sim 仿真环境。

```bash
cd /path/to/FluxVLA/FluxBisim
PYTHON_BIN=/path/to/install_dir/isaac-sim-4.5.0/python.sh ./fluxbisim_benchmark.sh pick_place_banana
```

终端 2：启动 FluxVLA 推理。

```bash
cd /path/to/FluxVLA
./scripts/fluxbisim_eval.sh \
  configs/pi05/fluxbisim/pi05_paligemma_pick_place_banana_full_finetune.py \
  /path/to/checkpoint.pth
```

注意：

- `scripts/fluxbisim_eval.sh` 会根据配置文件名映射任务编号，请确保配置与评测任务一致。

## 致谢

## 开源计划
