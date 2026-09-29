# P3：单机墙体绕障 PPO

P2 已验证固定墙前目标的物理、相机、backend、episode reset 和进程关闭生命周期。P3 将
最终目标放到 `contact_wall` 后方，先证明场景可达，再让 PPO 仅根据固定机身深度相机、
自身状态和最终目标学习绕墙。

## 训练边界

- PPO 输出仍只有 `vx_body、vy_body、vz_body、yaw_rate`；
- `decision_mask=0`，候选观测点选择头不参与本阶段训练；
- PPO 只收到墙后最终目标 `[6.0, 0.0, 1.5]`；
- `validation_waypoints_m` 仅由确定性可达性探针读取，不进入 observation、reward、训练
  rollout 或 checkpoint；
- 保留 P2 墙前目标 `[2.5, 0.0, 1.5]`，便于随时做回归。

## 同步文件

不要覆盖 Ubuntu 机器上的 `training/third_party`。同步以下文件：

```text
swarm_exploration/exploration_manager/config/rmappo_d455m_16.yaml
training/configs/isaac_single.yaml
training/training/racer_rmappo/isaac_scenarios.py
training/training/racer_rmappo/rule_navigation.py
training/training/racer_rmappo/isaac_single_backend.py
training/training/racer_rmappo/trainer.py
training/training/scripts/diagnose_isaac_wall_reachability.py
training/training/scripts/train_isaac_single_ppo.py
training/training/scripts/eval_isaac_single_ppo.py
training/training/tests/test_racer_rmappo.py
training/training/tests/test_isaac_single_backend.py
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

## 3. 1024 transition 连通性训练

首次运行只验证 PPO rollout、反向传播、checkpoint 和 Isaac 关闭链路，不用它判断策略
是否已经学会绕墙。单环境每次更新收集 256 transition，因此 1024 transition 是 4 次
PPO 更新。

```bash
PYTHONUNBUFFERED=1 python -u \
  training/scripts/train_isaac_single_ppo.py \
  --headless \
  --total-steps 1024 \
  --output runs/isaac_single_wall/ppo_smoke \
  2>&1 | tee runs/isaac_single_wall/ppo_smoke.log
echo ${PIPESTATUS[0]}
```

必须出现 `P3_ISAAC_PPO_TRAIN_RESULT=PASS`、最终 checkpoint 路径和退出码 0。训练指标应为
有限值；初始策略成功率低、碰撞率高并不代表链路失败。

## 4. 确定性评估

```bash
PYTHONUNBUFFERED=1 python -u \
  training/scripts/eval_isaac_single_ppo.py \
  runs/isaac_single_wall/ppo_smoke/checkpoint_final.pt \
  --headless \
  --steps 3200 \
  --output runs/isaac_single_wall/ppo_smoke_eval.json \
  2>&1 | tee runs/isaac_single_wall/ppo_smoke_eval.log
echo ${PIPESTATUS[0]}
```

`P3_ISAAC_PPO_EVAL_RUN_RESULT=PASS` 只表示评估程序正常完成。策略能力要看 JSON：

- `success_episode_rate`；
- `collision_free_success_rate`；
- `collision_episode_rate`；
- `timeout_episode_rate`；
- `stall_episode_rate`；
- `minimum_obstacle_clearance_m`；
- `episode_duration_s_mean`；
- `reward_mean`。

1024 transition 只做链路验收。确认链路稳定后，再以 checkpoint 续训到 5 万 transition，
每 5 千到 1 万 transition 做一次独立评估。固定场景达到至少 100 回合中无碰撞成功率
90%、碰撞率不高于 5%、超时率不高于 5%，才增加墙体尺寸、目标位置、树木和建筑随机化。

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
