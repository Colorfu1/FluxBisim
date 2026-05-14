# FluxBisim

`FluxBisim` 是基于 `Isaac Sim 4.5.0` 的机器人操作仿真环境，为 FluxVLA 提供仿真资产、数据采集脚本、控制器和 ROS 评测入口。

## 特性

- 支持 `Ubuntu 20.04` + `Isaac Sim 4.5.0`。
- 覆盖抓取放置、关箱、抽屉推拉、拧杯盖、书本传递等任务。
- 通过 ROS topic 连接 Isaac Sim 仿真环境与 FluxVLA 推理进程。
- 作为 `FluxVLA` 仓库下的子目录使用。

## 框架图

## 安装

### 1. 仓库目录

`FluxBisim` 应放在 `FluxVLA` 仓库目录下：

```text
FluxVLA/
├── FluxBisim/
├── configs/
├── scripts/
└── fluxvla/
```

`FluxBisim` 主要目录如下：

```text
FluxBisim/
├── assets/          # 仿真资产
├── controllers/     # 机器人控制器
├── data_collect/    # 数据采集脚本
├── envs/            # 仿真器、机器人和物体配置
├── tasks/           # 任务逻辑与随机化
├── evaluate.py      # Isaac Sim 评测入口
└── fluxbisim_benchmark.sh
```

### 2. 安装 Isaac Sim

下载并解压 Isaac Sim 4.5.0：

```bash
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip
unzip isaac-sim-standalone-4.5.0-linux-x86_64.zip -d /path/to/install_dir/
```

首次启动一次以初始化缓存和扩展：

```bash
cd /path/to/install_dir/isaac-sim-4.5.0
./isaac-sim.sh
```

建议为 Isaac Sim Python 配置别名：

```bash
echo 'alias benchmark_python="/path/to/install_dir/isaac-sim-4.5.0/python.sh"' >> ~/.bashrc
source ~/.bashrc
```

### 3. 安装 ROS Noetic

仿真评测依赖 ROS Noetic：

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

环境检查：

```bash
benchmark_python -c "import isaacsim, rootutils, yaml, h5py, scipy; print('FluxBisim env OK')"
```

## 数据采集

所有采集脚本均在 `FluxBisim/` 下运行：

```bash
cd /path/to/FluxVLA/FluxBisim
benchmark_python <collect_script> --env <env_name> --config <config_name> [--data_path <output_dir>]
```

可用任务：

| 任务         | 脚本                                                            | 环境                    | 配置                                                                                                                                                                                                           |
| ------------ | --------------------------------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 水果抓取放置 | `data_collect/pick_place_fruit/fruit_pick_place_collect.py`     | `kitchen`, `apartment`  | `apple_pick_place_config.yaml`, `banana_pick_place_config.yaml`, `carrot_pick_place_config.yaml`, `cucumber_pick_place_config.yaml`, `mangosteen_pick_place_config.yaml`, `whiteradish_pick_place_config.yaml` |
| 关箱         | `data_collect/close_box/box_close_collect.py`                   | `industry`              | `box_close_config.yaml`                                                                                                                                                                                        |
| 抽屉推拉     | `data_collect/pull_push_drawer/drawer_pull_push_collect.py`     | `apartment`             | `drawer_pull_push_config.yaml`                                                                                                                                                                                 |
| 拧杯盖       | `data_collect/rotate_pitcher_lid/pitcher_lid_rotate_collect.py` | `apartmentshort`        | `pitcher_lid_rotate_config.yaml`                                                                                                                                                                               |
| 书本传递     | `data_collect/pass_book/book_pass_collect.py`                   | `apartment`, `industry` | `book_pass_config.yaml`                                                                                                                                                                                        |

示例：

```bash
benchmark_python data_collect/pick_place_fruit/fruit_pick_place_collect.py \
  --env kitchen \
  --config banana_pick_place_config.yaml
```

## 训练

采集数据后，回到 `FluxVLA` 根目录启动训练：

```bash
cd /path/to/FluxVLA
bash scripts/train.sh \
  configs/pi05/fluxbisim/pi05_paligemma_pick_place_banana_full_finetune.py \
  work_dirs/pi05_paligemma_pick_place_banana_full_finetune
```

可通过 `--cfg-options` 覆盖训练参数：

```bash
bash scripts/train.sh \
  configs/pi05/fluxbisim/pi05_paligemma_pick_place_banana_full_finetune.py \
  work_dirs/pi05_paligemma_pick_place_banana_full_finetune \
  --cfg-options runner.max_steps=10000 runner.save_interval=1000
```

## 评测

支持的仿真评测任务：

```text
close_box
pass_book
pick_place_banana
pull_push_drawer
rotate_pitcher_lid
```

评测需要启动两个进程。

终端 1：启动 Isaac Sim 仿真环境。

```bash
cd /path/to/FluxVLA/FluxBisim
PYTHON_BIN=benchmark_python ./fluxbisim_benchmark.sh pick_place_banana \
  --num-episodes 10 \
  --save-images true \
  --chunk-size 32
```

终端 2：启动 FluxVLA 推理。

```bash
cd /path/to/FluxVLA
./scripts/fluxbisim_eval.sh \
  configs/pi05/fluxbisim/pi05_paligemma_pick_place_banana_full_finetune.py \
  /path/to/checkpoint.pth
```

注意：

- `fluxbisim_benchmark.sh` 会启动 `roscore`、`rviz` 和 `evaluate.py`。
- 如需指定 Isaac Sim Python，请通过 `PYTHON_BIN` 传入。
- `scripts/fluxbisim_eval.sh` 会根据配置文件名映射任务编号，请确保配置与评测任务一致。
