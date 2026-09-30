# P3：单机局部导航课程与墙体绕障验收

P2 已验证物理、固定机身 D455M、backend、episode reset 和进程关闭生命周期。
直接在墙后目标上训练到 10k transition 的策略在确定性评估中 8/8 回合碰撞，
说明“从零开始绕完整阻挡墙”课程过于突然。现改为同一 Isaac 环境下的三阶段训练：

| 场景 | 目标 | 用途 |
|---|---|---|
| `open_target` | `[2.0,-2.0,1.8]` | 目标跟踪、制动、yaw 和小高度变化 |
| `wall_edge` | `[5.0,-3.5,1.8]` | 沿墙南侧安全边缘通过，学习横向控制与近障减速 |
| `wall_avoidance` | `[6.0,0.0,1.5]` | 直线被墙完全阻断，作为后期固定绕障验收 |

本文档中的 P3 指“导航训练里程碑”；`ISAAC_RMAPPO_实施计划.md` 中的 P3 仍指
0.5 m 深度占据建图阶段，不要将两者混为已实现完整探索。

## 训练边界

- PPO 输出仍只有 `vx_body、vy_body、vz_body、yaw_rate`；
- `decision_mask=0`，候选观测点选择头不参与本阶段训练；
- PPO 每次只收到当前课程的一个局部目标；
- `--scenario` 自动选择匹配的 RMAPPO stage，不需要再手工配对 `--stage`；
- `validation_waypoints_m` 仅由确定性可达性探针读取，不进入 observation、reward、训练
  rollout 或 checkpoint；
- 保留 P2 墙前目标 `[2.5, 0.0, 1.5]`，便于随时做回归。
- 三个场景使用同一物理、相机、网络和 checkpoint 格式，不是三套一次性环境。

## 同步文件

不要覆盖 Ubuntu 机器上的 `training/third_party`。同步以下文件：

```text
swarm_exploration/exploration_manager/config/rmappo_d455m_16.yaml
training/configs/isaac_single.yaml
training/training/racer_rmappo/isaac_scenarios.py
training/training/racer_rmappo/trainer.py
training/training/scripts/train_isaac_single_ppo.py
training/training/scripts/eval_isaac_single_ppo.py
training/training/tests/test_racer_rmappo.py
training/P3_SINGLE_WALL_PPO.md
```

## 1. 纯 Python 回归

在没有 source Isaac 环境变量的新终端执行：

```bash
conda activate NavRL_lxc
cd /home/lww4090/racer_ws/src/RACER-main/training

env -u PYTHONPATH -u PYTHONHOME python -m pytest -q \
  training/tests/test_isaac_single_probe.py \
  training/tests/test_isaac_single_backend.py \
  training/tests/test_racer_rmappo.py
```

## 2. 确定性绕墙可达性验收

该步骤不是训练。控制器经过两个仅用于验收的侧向 waypoint，证明墙后目标在当前动力学、
边界和碰撞模型下可达。

```bash
conda activate NavRL_lxc
source "$ISAACSIM_PATH/setup_conda_env.sh"
cd /home/lww4090/racer_ws/src/RACER-main/training

set -o pipefail
PYTHONUNBUFFERED=1 python -u \
  training/scripts/diagnose_isaac_wall_reachability.py \
  --headless \
  --output runs/isaac_single_wall/reachability_report.json \
  2>&1 | tee runs/isaac_single_wall/reachability_console.log
status=${PIPESTATUS[0]}
echo "EXIT_CODE=$status"

grep -Ec \
  'PxArticulation(Link::setGlobalPose|ReducedCoordinate::setRootGlobalPose)' \
  runs/isaac_single_wall/reachability_console.log || true
```

进入训练前必须满足：

- `P3_WALL_REACHABILITY_RESULT=PASS` 且退出码为 0；
- 两个 waypoint 均到达，最终 `success=true`；
- `collision=false、out_of_bounds=false、timeout=false、stall=false`；
- `wall_observed=true`；
- pose 错误只允许是已记录的启动阶段单次兼容告警，不能随 waypoint/episode 增长。

## 3. 阶段 A：空旷局部目标

不要从已经学会撞墙的 `ppo_10k` checkpoint 恢复。先用新输出目录做 1024 transition
连通性训练：

```bash
mkdir -p runs/isaac_nav_curriculum

PYTHONUNBUFFERED=1 python -u \
  training/scripts/train_isaac_single_ppo.py \
  --scenario open_target \
  --headless \
  --total-steps 1024 \
  --output runs/isaac_nav_curriculum/open \
  2>&1 | tee runs/isaac_nav_curriculum/open_smoke.log
status=${PIPESTATUS[0]}
echo "EXIT_CODE=$status"
```

链路通过后在同一目录恢复到累计 10240 transition：

```bash
PYTHONUNBUFFERED=1 python -u \
  training/scripts/train_isaac_single_ppo.py \
  --scenario open_target \
  --headless \
  --resume runs/isaac_nav_curriculum/open/checkpoint_final.pt \
  --total-steps 10240 \
  --output runs/isaac_nav_curriculum/open \
  2>&1 | tee runs/isaac_nav_curriculum/open_10k.log
```

## 4. 阶段 A 确定性评估

```bash
PYTHONUNBUFFERED=1 python -u \
  training/scripts/eval_isaac_single_ppo.py \
  runs/isaac_nav_curriculum/open/checkpoint_final.pt \
  --scenario open_target \
  --headless \
  --steps 1600 \
  --progress-interval 100 \
  --output runs/isaac_nav_curriculum/open_eval.json \
  2>&1 | tee runs/isaac_nav_curriculum/open_eval.log
```

`P3_ISAAC_PPO_EVAL_RUN_RESULT=PASS` 只表示评估程序正常完成。策略能力要看 JSON：
评估默认每 100 个控制步输出一次 `[eval]`、已完成回合数和 ETA；单相机后端
速度约为 10 control step/s 时，1600 步大约需要 2～3 分钟。长时间没有新的 `[eval]`
输出才应按实际卡死排查。

- `success_episode_rate`；
- `collision_free_success_rate`；
- `collision_episode_rate`；
- `timeout_episode_rate`；
- `stall_episode_rate`；
- `minimum_obstacle_clearance_m`；
- `actual_speed_mps_mean` 和 `actual_speed_mps_max`；
- `minimum_target_distance_m` 和 `final_target_distance_m`；
- `episode_duration_s_mean`；
- `reward_mean`。

阶段 A 晋级建议：`episode_count>=20`、无碰撞成功率至少 90%、碰撞率不高于 2%、
停滞率不高于 5%。样本不足 20 回合时延长 `--steps`，不用小样本的 0% 冒充通过。

## 5. 阶段 B：沿墙边缘通过

阶段 A 通过后，恢复其 checkpoint。`--total-steps` 是包含前阶段的累计目标：

```bash
PYTHONUNBUFFERED=1 python -u \
  training/scripts/train_isaac_single_ppo.py \
  --scenario wall_edge \
  --headless \
  --resume runs/isaac_nav_curriculum/open/checkpoint_final.pt \
  --total-steps 30720 \
  --output runs/isaac_nav_curriculum/wall_edge \
  2>&1 | tee runs/isaac_nav_curriculum/wall_edge_30k.log

PYTHONUNBUFFERED=1 python -u \
  training/scripts/eval_isaac_single_ppo.py \
  runs/isaac_nav_curriculum/wall_edge/checkpoint_final.pt \
  --scenario wall_edge \
  --headless --steps 2000 --progress-interval 100 \
  --output runs/isaac_nav_curriculum/wall_edge_eval.json \
  2>&1 | tee runs/isaac_nav_curriculum/wall_edge_eval.log
```

阶段 B 晋级建议：至少 20 回合，无碰撞成功率至少 80%，碰撞率不高于 5%，
且 `minimum_obstacle_clearance_m` 不应长期贴在 0.9 m 的相机最小量程。

## 6. 阶段 C：墙后目标

```bash
PYTHONUNBUFFERED=1 python -u \
  training/scripts/train_isaac_single_ppo.py \
  --scenario wall_avoidance \
  --headless \
  --resume runs/isaac_nav_curriculum/wall_edge/checkpoint_final.pt \
  --total-steps 81920 \
  --output runs/isaac_nav_curriculum/wall_avoidance \
  2>&1 | tee runs/isaac_nav_curriculum/wall_avoidance_80k.log

PYTHONUNBUFFERED=1 python -u \
  training/scripts/eval_isaac_single_ppo.py \
  runs/isaac_nav_curriculum/wall_avoidance/checkpoint_final.pt \
  --scenario wall_avoidance \
  --headless --steps 3200 --progress-interval 100 \
  --output runs/isaac_nav_curriculum/wall_avoidance_eval.json \
  2>&1 | tee runs/isaac_nav_curriculum/wall_avoidance_eval.log
```

固定墙后场景最终验收为至少 100 完整回合中：无碰撞成功率至少 90%、碰撞率
不高于 5%、超时率不高于 5%。只有这项通过后，才开始目标位置、高度、墙体、
树木和建筑的程序化随机。

## 结果驱动调整

- 几乎全是碰撞：先看 `near_obstacle` 奖励和深度图是否随 yaw 正确变化；不要先增大
  `collision_terminal`。
- 几乎全是 stall：检查动作均值/标准差和实际速度，再考虑延长 stall 窗口；不能直接关闭
  stall 终止掩盖零动作策略。
- reward 上升但成功率不升：通常是策略利用进度奖励在墙前往返，应减少局部奖励漏洞或增加
  成功奖励，而不是仅观察总 reward。
- 成功但频繁擦墙：提高 held-out 评估中对最小 clearance 的要求，再调整 near-obstacle
  安全距离或权重。
- 训练速度过慢：当前单相机约 10 control step/s。先分析 RTX 相机 capture；不要直接复制
  成 16 个 640x360 render product。
