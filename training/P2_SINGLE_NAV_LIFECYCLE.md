# P2 第三步：单机成功回合与 reset 生命周期

固定目标 backend 合同通过后，本探针用确定性比例控制器连续完成 20 个回合。它不训练
PPO，目标是验收以下闭环：

```text
规则动作 -> 到达目标 -> observation_completed -> success/done
 -> 自动 reset -> Fabric 相机位姿同步 -> 新深度栈 -> 下一回合
```

物理探针构造完成时已经位于初始状态，因此 backend 第一次 reset 直接消费该状态，避免
Isaac Sim 2023.1 在启动阶段连续执行两次相同的 Direct-GPU tensor teleport。以后每次
episode reset 仍执行正常 tensor reset，并执行一个零指令控制周期，使 PhysX 的根位姿
传播给固定在 `base_link` 下的相机，再将速度清零。探针既对比初始与每次 reset 后的
inverse-depth，也用相机光轴和前方墙体几何计算绝对中心深度，避免两张相同的错误图像
被误判为正常。

## 同步文件

```text
training/configs/isaac_single.yaml
training/training/racer_rmappo/isaac_adapter.py
training/training/racer_rmappo/isaac_runtime.py
training/training/racer_rmappo/isaac_single_env.py
training/training/racer_rmappo/isaac_single_backend.py
training/training/racer_rmappo/rule_navigation.py
training/training/scripts/diagnose_isaac_navigation_backend.py
training/training/scripts/diagnose_isaac_navigation_lifecycle.py
training/training/tests/test_isaac_single_backend.py
training/P2_SINGLE_NAV_LIFECYCLE.md
```

不要覆盖 Ubuntu 上的 `training/third_party`。

## 执行

先在未 source Isaac 的终端运行纯 Python 回归：

```bash
conda activate NavRL_lxc
cd /home/lww4090/racer_ws/src/RACER-main/training
python -m pytest -q \
  training/tests/test_isaac_single_probe.py \
  training/tests/test_isaac_single_backend.py \
  training/tests/test_racer_rmappo.py
```

再在新的终端运行 Isaac 生命周期探针，并保存完整日志：

```bash
conda activate NavRL_lxc
source "$ISAACSIM_PATH/setup_conda_env.sh"
cd /home/lww4090/racer_ws/src/RACER-main/training

set -o pipefail
PYTHONUNBUFFERED=1 python -u \
  training/scripts/diagnose_isaac_navigation_lifecycle.py \
  --episodes 20 \
  --headless \
  --output runs/isaac_single_backend/lifecycle_report.json \
  2>&1 | tee runs/isaac_single_backend/lifecycle_console.log
echo ${PIPESTATUS[0]}

grep -Ec 'PxArticulation(Link::setGlobalPose|ReducedCoordinate::setRootGlobalPose)' \
  runs/isaac_single_backend/lifecycle_console.log
```

第一次同步关闭修复后，先用 `--episodes 2` 做快速退出验收。只有日志依次出现
`ISAAC_REPLICATOR_DRAINED`、`ISAAC_USD_STAGE_CLOSED`、`Simulation App Shutting Down`，
`Simulation App Shutdown Complete`、`ISAAC_NATIVE_SHUTDOWN_RETURNED`，且 shell 退出码为
0，才运行正式 20 回合。功能报告显示 PASS 但随后段错误、退出码 139，仍算失败。

`hard_exit_after_shutdown=true` 只跳过 Kit 已完成原生关闭之后的 Python 解释器终结清理；
它不会跳过 backend、Replicator、USD stage 或 SimulationApp 的关闭，也会保留异常时的
退出码 1/2。该兼容路径是针对锁定的 Isaac Sim 2023.1，不应无条件复制到未来版本。

预期 `P2_LIFECYCLE_RESULT=PASS`、`P2_LIFECYCLE_ENV_CLOSED`、退出码 0，并满足：

- 20/20 回合成功；
- 碰撞、越界、超时和停滞回合均为 0；
- reset 位置误差不超过 0.005 m、速度不超过 0.001 m/s；
- reset yaw 误差不超过 0.01 rad；
- reset depth MAE 不超过 0.02；
- 初始及各次 reset 的 actor 中心 inverse-depth 与配置墙面几何值误差不超过 0.03；
- reset 后三张历史深度帧完全来自同一次新观测，帧间最大误差不超过 `1e-6`；
- reset 期间无碰撞、越界或非有限状态；
- Torch CUDA reserved memory 增长不超过 256 MiB。

Torch 内存只覆盖 PyTorch allocator，不包含完整 Kit/RTX 显存。正式长训练前还要另外观察
`nvidia-smi` 的进程显存是否随回合持续线性增长。

## Direct GPU API 错误判读

不要过滤 `PxArticulationLink::setGlobalPose()` 或
`PxArticulationReducedCoordinate::setRootGlobalPose()` 日志。本探针将其出现次数和
JSON 一起保留：

- 计数为 0：reset 数据与日志均通过，可以进入 Isaac PPO 训练入口开发。
- 只在初始化出现一次、20 次 reset 均正确：记录为 Isaac 2023.1 初始化兼容问题，仍需
  在更长压力测试中观察。
- 随每个回合增长，或同时出现 reset 位置/深度错误：停止进入 PPO，优先隔离
  flatcache、GPU pipeline 和相机子 prim 的同步路径。

规则比例控制器仅用于验证固定目标可完成，不能作为避障 teacher，也不能证明 PPO 已经
学会导航。
