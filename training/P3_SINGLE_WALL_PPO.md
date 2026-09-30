# P3：单机局部导航课程与墙体绕障验收

2026-09-30 更新：`open_target` 10k 评估尚未通过，先按本文末尾“评估可视化与逐步诊断”
检查目标附近运动与超速，再决定续训；暂不进入 wall_edge。

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

## 评估可视化与逐步诊断（2026-09-30）

本次 open_target 日志：5 个完整回合、0 成功/0 碰撞、3 超时、1 停滞、1 越界；距离最小
0.6165 m，大于 0.5 m 到达阈值。已学到接近目标，但不能据此认定学会到达、制动或避障。
后期 approx_kl 多次为 0.05～0.08，最大实际速度 3.873 m/s；需要先用轨迹证据定位，
不宜仅增加步数或放宽到达阈值。2 m/s 当前约束速度指令，不能保证动力学实际速度没有超调。

新增文件/需要同步的改动：

```text
training/training/racer_rmappo/eval_visualization.py
training/training/racer_rmappo/isaac_single_backend.py
training/training/racer_rmappo/trainer.py
training/training/scripts/eval_isaac_single_ppo.py
training/training/scripts/plot_isaac_eval_trace.py
training/training/tests/test_eval_visualization.py
training/training/tests/test_isaac_single_backend.py
training/P3_SINGLE_WALL_PPO.md
```

本次只增加可选评估诊断，不改变奖励、策略输入、checkpoint 格式或控制器。
在未 source Isaac 的终端执行回归（路径相对于 RACER-main/training）：

```bash
python training/tests/test_eval_visualization.py
env -u PYTHONPATH -u PYTHONHOME python -m pytest -q \
  training/tests/test_isaac_single_backend.py training/tests/test_racer_rmappo.py
```

在 Ubuntu 本机桌面终端，激活 NavRL_lxc 并 source Isaac 后运行已有 checkpoint：

```bash
cd /home/lww4090/racer_ws/src/RACER-main/training
mkdir -p runs/isaac_nav_curriculum
set -o pipefail
PYTHONUNBUFFERED=1 python -u training/scripts/eval_isaac_single_ppo.py \
  runs/isaac_nav_curriculum/open/checkpoint_final.pt \
  --scenario open_target --no-headless --visualize --view overview \
  --steps 1600 --progress-interval 100 --draw-every 5 \
  --trace runs/isaac_nav_curriculum/open_visual.trace.jsonl \
  --output runs/isaac_nav_curriculum/open_visual_eval.json \
  2>&1 | tee runs/isaac_nav_curriculum/open_visual_eval.log
eval_exit=${PIPESTATUS[0]}
echo "EXIT_CODE=$eval_exit"
```

GUI 中：灰色实体是实际静态障碍；青色线是轨迹；绿色球是目标位置容差；紫色向量表示
实际世界系速度，绿色向量是经过限幅、加速度和边界过滤后的世界系速度指令（长度按 1 s
位移显示）；黄色是随完整机身姿态转动的几何视锥，为易读只画 3 m，传感器上限仍为 20 m。
可选 `--view top` 俯视或 `--view follow` 跟随；HUD 显示距离、yaw、倾斜、速度、奖励与结束原因。
GUI 结束时自动关闭；录制视频可使用桌面录屏，本次未增加自动 MP4 导出。

线条通过 Isaac 2023.1 的 `omni.isaac.debug_draw` 绘制，不创建碰撞体或 USD 目标实体。
只有图形刷新限频，JSONL 每个控制步均保存；最多在窗口保留 4000 条轨迹段。
回合终止会自动 reset，HUD/轨迹记录的是 reset 前状态，画面中的机体可能已回到出生点；
不会把旧终点和新起点连成飞行轨迹。GUI 比 headless 更慢，应分别记录吞吐。

没有桌面/仅 SSH 时，使用 headless 轨迹记录和离线图片：

```bash
PYTHONUNBUFFERED=1 python -u training/scripts/eval_isaac_single_ppo.py \
  runs/isaac_nav_curriculum/open/checkpoint_final.pt \
  --scenario open_target --headless --steps 1600 --progress-interval 100 \
  --trace runs/isaac_nav_curriculum/open_trace.jsonl \
  --output runs/isaac_nav_curriculum/open_trace_eval.json

# 可在另一个装有 matplotlib 的普通 Python 环境运行，无需 Isaac/ROS/Torch。
python training/scripts/plot_isaac_eval_trace.py \
  runs/isaac_nav_curriculum/open_trace.jsonl \
  --output runs/isaac_nav_curriculum/open_diagnostics.png
```

图片包含 XY 场景/轨迹、目标距离、实际/指令速度、高度、yaw/倾斜与逐步奖励。
XY 障碍是场景真值投影；它不是机器人已知地图。JSONL 首行为解析后的配置，后续每行为
一个终止前的控制步快照，包括机身四元数、请求速度、过滤后速度、积分位置参考、分项奖励和深度有效比例。
JSON 最终报告的 `diagnostics` 还给出超过 2.05 m/s 的控制步数和最大速度那一步的完整记录。
这不是物理子步最大速度的保证；需更细分析时再提升物理子步记录频率。

重点检查：

- 距离始终大于 0.5 m：先看过冲、横向/高度偏差，不能归因于目标 yaw。
- 距离已合格但未完成：对照 yaw 0.25 rad 与 tilt 0.20 rad 阈值。
- 实际速度大于 2 m/s、过滤后指令小于等于 2 m/s：检查控制跟踪误差和积分位置参考累积；
  不要把截断速度统计值当作修复。
- 越界发生在哪个轴、之前是否出现姿态和速度异常，查看终止行与 peak_speed_transition。
- KL 连续升高：结合 metrics.jsonl 中 clip_fraction、entropy、policy/value loss 和梯度范数检查；
  若更新仍过大，可试 lr=1e-4、epochs=3，但恢复 optimizer 会带回 checkpoint 的学习率，
  不能只改 YAML 就假定新学习率已经生效。

当前 Isaac backend 没有占据建图/frontier，coverage=0 是占位值。视锥不考虑墙体遮挡，
也不表示任何体素已经被观测；不会把飞行轨迹或整个视锥填成“探索区域”。后续接入 0.5 m
深度射线融合后，才能真实绘制 unknown/free/occupied、frontier、任务区以及团队已观测区域。
首轮可视化验收还需比对同 checkpoint/场景的 GUI 与 headless 轨迹和深度有效比例，确保
叠加显示未污染传感器或改变 reset 时序；不要求 GPU 仿真逐位一致。
