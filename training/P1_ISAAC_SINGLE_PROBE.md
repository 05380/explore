# P1 第一阶段：单机 Isaac 物理探针

这一阶段只验证 Isaac Sim、OmniDrones、Hummingbird、Lee 控制器、碰撞检测和 reset 是否形成可靠闭环。它不读取深度图，不构造体素地图，不运行 PPO，也不代表已经具备避障能力。

## 已实现内容

- `configs/isaac_single.yaml`：30×30×5 m 单机场景、2 m/s 速度参考上限、2 m/s² 加速度上限、静态障碍和验收阈值。
- `racer_rmappo/isaac_single_env.py`：通过 Isaac Core/USD 原生 API 创建本地地面与障碍，不依赖可选的 `omni.isaac.orbit` 扩展；包含 Hummingbird、航向局部系速度/偏航指令及限幅、Lee 控制器、状态/接触力/越界遥测、完整 reset。动作约定为 x 向机头前方、y 向左、z 沿世界向上。
- `scripts/diagnose_isaac_backend.py`：独立启动和关闭 `SimulationApp`，执行探针并保存 JSON 报告。
- `tests/test_isaac_single_probe.py`：无需 Isaac 的配置与动作限幅回归测试。

该实现有意没有把 `trainer.py --backend isaac` 打开。当前环境还缺少 D455M 深度观测和完整 `MultiUAVBackend` 合同，提前接入会让无效的占位观测进入 PPO。

## 同步到 Ubuntu

只同步本次新增或修改的文件，不要再次整体替换 `training/third_party/tensordict` 或 `training/third_party/rl`，否则会删除刚编译的 `_tensordict*.so` 和 `_torchrl.so`。

需要同步：

```text
training/configs/isaac_single.yaml
training/training/racer_rmappo/isaac_single_env.py
training/training/scripts/diagnose_isaac_backend.py
training/training/tests/test_isaac_single_probe.py
training/P1_ISAAC_SINGLE_PROBE.md
training/README_RMAPPO.md
ISAAC_RMAPPO_实施计划.md
```

## 执行顺序

先在不加载 Isaac 的新终端执行纯 Python 测试：

```bash
conda activate NavRL_lxc
cd /home/lww4090/racer_ws/src/RACER-main/training
python -m pytest -q \
  training/tests/test_isaac_single_probe.py \
  training/tests/test_racer_rmappo.py \
  training/tests/test_wall_teacher.py
```

再打开一个新终端加载 Isaac 环境。`SimulationApp` 启动后不要再运行 conda 子命令：

```bash
conda activate NavRL_lxc
source "$ISAACSIM_PATH/setup_conda_env.sh"
cd /home/lww4090/racer_ws/src/RACER-main/training

PYTHONUNBUFFERED=1 python -u training/scripts/diagnose_isaac_backend.py \
  --config configs/isaac_single.yaml \
  --probe all \
  --headless \
  --output runs/isaac_single_probe/report.json
```

成功标志为：

```text
P1_PROBE_RESULT=PASS
```

结果同时保存在：

```text
runs/isaac_single_probe/report.json
runs/isaac_single_probe/resolved_config.yaml
```

脚本会直接从 `third_party/OmniDrones` 加载项目内版本，不要求预先执行 `pip install -e`。如果初始化期间发生异常，`report.json` 仍会写入异常类型、错误消息和 traceback；`resolved_config.yaml` 仅在环境初始化并运行探针后生成。

若全量探针失败，用单项命令缩小范围：

```bash
python -u training/scripts/diagnose_isaac_backend.py --probe hover --steps 300 --headless
python -u training/scripts/diagnose_isaac_backend.py --probe reset --headless
python -u training/scripts/diagnose_isaac_backend.py --probe random --steps 300 --headless
python -u training/scripts/diagnose_isaac_backend.py --probe contact --headless
```

## 如何读结果并调整

### hover

主要字段：

- `finite` 必须为 true；否则先查 PyTorch/Isaac ABI、物理步长和控制器输出，不能调奖励。
- `collision_steps` 和 `out_of_bounds_steps` 必须为 0。
- `max_position_error_m` 检查起飞瞬态，默认不超过 0.50 m。
- `final_position_error_m` 检查稳态误差，默认不超过 0.20 m。
- `min_up_z` 默认不低于 0.80；下降通常表示姿态控制不稳定。

如果持续上下振荡，先把 `sim.physics_dt` 从 0.016 改为 0.01；如果稳定但误差略高，再检查 Hummingbird 参数与 Lee 控制器是否匹配。不要第一反应放宽验收阈值。

### reset

主要字段是 `max_position_error_m` 和 `max_speed_mps`，默认均应接近 0。如果失败，检查 `set_world_poses`、`set_velocities`、`drone._reset_idx` 和 physics view `flush` 的顺序。reset 没通过前不能采集 rollout，否则不同回合会相互污染。

### random

- `requested_max_speed_mps` 应故意超过 2 m/s，以证明测试覆盖了限幅。
- `limited_max_speed_mps` 必须不超过 2 m/s。
- `limited_max_yaw_rate_rps` 必须不超过 1 rad/s。
- `velocity_tracking_rmse_mps` 衡量 Lee 控制器对速度参考的跟踪。
- `actual_max_speed_mps` 是实际物理速度，不等于策略指令限幅。
- `actual_overspeed_steps` 和 `actual_overspeed_fraction` 统计真实速度超过 2 m/s 的持续程度；瞬态超调必须与 `actual_max_speed_mps` 一起判断。
- `collision_steps` 在随机无避障指令下可以大于 0，本阶段只记录，不把它单独判为失败。
- `sim_steps_per_second` 和 `real_time_factor` 是未启用深度相机时的性能基线；下一阶段加入相机后要用相同并行数、步数和 headless 设置复测。

速度参考会先经过 2 m/s² 变化率限制，并在接近飞行边界时按制动距离削减外向速度。这是飞行包线约束，不替 PPO 做障碍避让。如果 RMSE 仍高但悬停稳定，先把 `random_command_interval_steps` 从 125 增加到 200，判断是否只是指令切换过快；之后再考虑速度控制器增益。若实际速度仍明显超过 3 m/s，先检查参考位置积分与控制器增益，而不是修改 PPO 或放宽验收阈值。

### contact

- `detected` 必须为 true，证明故意把无人机置入第一个障碍后能读到接触力。
- `cleared_after_reset` 必须为 true，证明碰撞状态不会污染下一回合。
- `max_contact_force_n` 应高于 `contact_threshold_n`。

若接触力始终为 0，检查 `drone.initialize(track_contact_forces=True)`、障碍的 collision schema，以及无人机 base link 的 contact view。若无碰撞时偶发误报，先观察静止噪声分布，再小幅提高 `contact_force_threshold_n`，并保留安全余量。

## 通过后的后续开发

1. 给机身 `base_link` 挂载固定 D455M 相机，先读取原始 `distance_to_camera`。
2. 验证相机位姿随无人机 yaw 转动、相机本身没有独立转动自由度。
3. 明确深度语义并裁剪到 0.3～20 m，记录无效像素比例、帧率和 GPU 显存。
4. 反投影深度视锥并维护 0.5 m 本机占据/已观测体素。
5. 用真实深度、ego、固定局部目标构造单机版 `MultiUAVBackend`；此时才打开 `--backend isaac`。
6. 先做固定目标和固定障碍的 4096-step 连通性训练，再做固定场景过拟合；成功后才随机化建筑和树木。
