# limx_bench

本项目基于 `Isaac Sim 4.5.0`，提供单臂/双臂操作任务的数据采集脚本与控制器实现，当前支持的操作系统为 `Ubuntu 20.04`。

## Isaac Sim 4.5.0 安装与环境配置

推荐用户通过 NVIDIA 官方直链下载 `Isaac Sim 4.5.0` 安装包：

[Isaac Sim 4.5.0 Linux x86_64 下载链接](https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip)

### 1. 安装 Isaac Sim 4.5.0

请先在 `Ubuntu 20.04` 上下载并解压 `Isaac Sim 4.5.0`。例如：

```bash
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip
unzip isaac-sim-standalone-4.5.0-linux-x86_64.zip -d /path/to/install_dir/
```

解压完成后，目录通常类似：

```bash
/path/to/install_dir/isaac-sim-4.5.0
```

建议首次启动一次，完成缓存和扩展初始化：

```bash
cd /path/to/install_dir/isaac-sim-4.5.0
./isaac-sim.sh
```

### 2. 配置 Isaac Sim 自带 Python 环境

`Isaac Sim` 自带一个定制的 Python 解释器，入口脚本为 `python.sh`。建议在 `~/.bashrc` 中添加别名：

```bash
alias benchmark_python="/path/to/install_dir/isaac-sim-4.5.0/python.sh"
```

配置生效后可以这样使用：

```bash
source ~/.bashrc
benchmark_python -c "import isaacsim; print('Isaac Sim Python OK')"
benchmark_python <python_file_name>
```

## 项目安装与运行

### 1. 获取本仓库源码

```bash
git clone <your-repo-url>
cd limx_bench
```

仓库结构如下：

```text
limx_bench
├── assets          # 场景、机器人、物体相关资产
├── controllers     # 机械臂运动控制器程序
├── data_collect    # 任务数据采集脚本
├── envs            # 仿真器、机器人、物体配置
└── tasks           # 任务随机化与测评逻辑
```

### 2. 安装本仓库额外依赖

本项目代码中额外使用了 `rootutils`、`PyYAML`、`h5py`、`scipy` 等 Python 包，建议直接安装到 Isaac Sim 自带环境中：

```bash
benchmark_python -m pip install --upgrade pip
benchmark_python -m pip install rootutils PyYAML h5py scipy
```

如果你没有配置别名，也可以直接执行：

```bash
/path/to/install_dir/isaac-sim-4.5.0/python.sh -m pip install rootutils PyYAML h5py scipy
```

### 3. 验证环境是否可用

在项目根目录执行：

```bash
benchmark_python -c "import rootutils, yaml, h5py, scipy; print('Python deps OK')"
benchmark_python -c "import isaacsim; print('Isaac Sim import OK')"
```

如果两条命令都能正常输出，说明运行环境已经准备完成。

### 4. 数据采集

所有采集脚本都建议在项目根目录执行，并统一使用以下参数：

- `--env`：环境名称，脚本会自动映射到 `<env>_table` 和 `base_<env>`
- `--config`：当前任务目录下的配置文件名
- `--data_path`：可选，自定义数据输出路径

#### 水果抓取与放置（厨房/客厅）

```bash
benchmark_python data_collect/pick_place_fruit/fruit_pick_place_collect.py \
  --env kitchen \
  --config apple_pick_place_config.yaml
```

可选环境：

- `kitchen`
- `apartment`

可选配置：

- `apple_pick_place_config.yaml`
- `banana_pick_place_config.yaml`
- `carrot_pick_place_config.yaml`
- `cucumber_pick_place_config.yaml`
- `mangosteen_pick_place_config.yaml`
- `whiteradish_pick_place_config.yaml`

#### 零件放置与箱子闭合（工业）

```bash
benchmark_python data_collect/close_box/box_close_collect.py \
  --env industry \
  --config box_close_config.yaml
```

可选环境：

- `industry`

可选配置：

- `close_box_config.yaml`

#### 抽屉抽拉（客厅）

```bash
benchmark_python data_collect/pull_push_drawer/drawer_pull_push_collect.py \
  --env apartment \
  --config drawer_pull_push_config.yaml
```

可选环境：

- `apartment`

可选配置：

- `drawer_pull_push_config.yaml`

#### 拧杯盖（客厅矮桌）

```bash
benchmark_python data_collect/rotate_pitcher_lid/pitcher_lid_rotate_collect.py \
  --env apartmentshort \
  --config pitcher_lid_rotate_config.yaml
```

可选环境：

- `apartmentshort`

可选配置：

- `pitcher_lid_rotate_config.yaml`

#### 书本传递（客厅/工业）

```bash
benchmark_python data_collect/pass_book/book_pass_collect.py \
  --env apartment \
  --config book_pass_config.yaml
```

可选环境：

- `apartment`
- `industry`

可选配置：

- `book_pass_config.yaml`