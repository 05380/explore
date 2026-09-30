# 规则选点 + 四维 PPO 导航（2026-09-30）

本页是当前训练与验收入口，优先于历史九维混合策略说明。无需 ROS 或 catkin 编译。

## 本次改动与边界

- `navigation_v2`：CNN + GRU，actor 只读 `depth/ego/target/neighbors`，只输出归一化
  `[vx_forward, vy_left, vz_up, yaw_rate]`；位置与 yaw 的目标均由规则提供。
- `hybrid_v1`：只保留旧 checkpoint 的评估与导出，不允许续训；旧文件无版本字段时须包含
  已知候选头参数才能识别。新旧网络不能混载，新导航网络从头训练。
- 物理始终 `step(render=False)`，每控制动作执行 6 次控制器/推力/物理更新。
  相机负责 render-only 刷新，叠加图只更新绘图数据；物理回调实测审计失败立即报错。
- 默认不改控制器、到达阈值和导航奖励量级。新策略关闭候选增益、本机/团队新体素、重复体素
  和覆盖里程碑奖励；历史 YAML 中保留的相关系数仅供旧版本回放，不在 v2 生效。
- 规则选点器、目标接口已实现；Isaac 仍使用 `FixedGoalProvider` 提供课程点。真实占据融合、
  frontier 候选生成、地图通信和多机任务分配尚未接入，不能称为完整探索。
- 原生 Isaac/GUI 需要在 Ubuntu GPU 机器验收；CPU 单测不能证明本次下坠已在真实仿真修好。

旧日志中 133 个完整回合全部在名义 0.6 s 内下降越界。上升指令与实际下坠矛盾，GUI 与 headless
 步进分支是待验证原因；目前不据此调整 PPO 奖励或断言训练失败。`RUN_RESULT=PASS` 仅指程序运行完成。

### 本地验证记录

2026-09-30 在隔离的 macOS CPU 环境（PyTorch 2.2.2、NumPy 1.26.4）完成：

- 下列五个测试文件合计 **52 passed**，覆盖物理步进 mock、规则选择、目标切换、循环策略、旧模型兼容及可视化数据。
- CLI smoke 实际运行 512 transitions，保存 v2 checkpoint；随后完成加载评估和 TorchScript 导出。
- 导出 actor 在 `[E,A]=(1,1)、(2,4)、(1,16)` 输入下输出四维有限动作及各机独立 hidden 张量。
- `git diff --check` 通过。

这些是软件接口验证，不是原生 Isaac、真实多机仿真或策略成功率验收。Ubuntu 三模式物理一致性、传感器、20 次 reset 及新策略课程成绩仍待实测。

## 1. 同步与轻量回归

保留 Ubuntu 的 `training/runs/`（模型）和 `training/third_party/`（已编译依赖）。同步本次变动的
`training/training/racer_rmappo/`、`training/training/scripts/`、`training/training/tests/`、文档，以及
`swarm_exploration/exploration_manager/config/rmappo_d455m_16.yaml`。不要再次整体覆盖模型与第三方目录。

在尚未 source Isaac 的新终端中：

```bash
conda activate NavRL_lxc
cd /home/lww4090/racer_ws/src/RACER-main/training
env -u PYTHONPATH -u PYTHONHOME python -m pytest -q \
  training/tests/test_navigation_v2.py \
  training/tests/test_isaac_single_backend.py \
  training/tests/test_isaac_single_probe.py \
  training/tests/test_racer_rmappo.py \
  training/tests/test_eval_visualization.py
```

CPU smoke 可检查更新、保存、评估、导出，但不训练真实避障。现有 NavRL_lxc 已有依赖时无需重装：

```bash
python training/scripts/train_rmappo.py --backend smoke --device cpu \
  --num-envs 2 --total-steps 1024 --output /tmp/racer_navigation_v2_smoke
python training/scripts/eval_rmappo.py /tmp/racer_navigation_v2_smoke/checkpoint_final.pt \
  --device cpu --num-envs 2 --steps 256
python training/scripts/export_rmappo_actor.py /tmp/racer_navigation_v2_smoke/checkpoint_final.pt \
  /tmp/racer_navigation_v2_smoke/actor.ts
```

若输出目录已有 checkpoint，使用新的目录或显式 `--resume`，不要删除旧模型来解决冲突。

## 2. 先验收物理：同一配置三种模式

在上面的已激活环境中，且仅为 Isaac 命令加载运行环境：

```bash
source "$ISAACSIM_PATH/setup_conda_env.sh"
python -u training/scripts/diagnose_isaac_timing.py --headless \
  --output runs/timing_v2/headless.json
python -u training/scripts/diagnose_isaac_timing.py --no-headless \
  --output runs/timing_v2/gui.json
python -u training/scripts/diagnose_isaac_timing.py --no-headless --visualize \
  --output runs/timing_v2/gui_overlay.json
python training/scripts/diagnose_isaac_timing.py \
  --compare runs/timing_v2/headless.json runs/timing_v2/gui.json runs/timing_v2/gui_overlay.json \
  --output runs/timing_v2/comparison.json
```

GUI 命令必须在有可用图形桌面的终端运行，不要手动按 Play/Stop/Space 改变时间线。
每次为 600 步悬停（30 s）+ 200 步固定速度序列（10 s），三个模式都采集深度。

验收要求：

- 各报告 `passed=true`；`physics_timing.valid=true`、`render_physics_steps=0`。
- 每动作 `timing_step.physics_steps=6`、`controller_updates=6`、`force_applications=6`、
  `physics_time_s≈0.05`。累计时间来自 PhysX 回调，reset 同步步数另列。
- 悬停最大位置误差 ≤0.5 m、结束误差 ≤0.2 m，无碰撞/越界。
- 比较报告位置差 ≤0.05 m、速度差 ≤0.1 m/s；不要求 GPU 逐比特一致。
- 如果回调没有触发，也会失败；不要关闭审计来绕过它，应核查安装版本的回调接口。

然后重复已有完整探针及 20 回合生命周期测试：

```bash
python -u training/scripts/diagnose_isaac_backend.py --probe all --headless \
  --output runs/timing_v2/full_regression.json
python -u training/scripts/diagnose_isaac_navigation_lifecycle.py --headless \
  --output runs/timing_v2/lifecycle.json
```

若失败，先发 `comparison.json`、失败模式的 JSON 和第一次异常前后的日志。
如果时序通过但仍下坠，比较 `rotor_action/rotor_thrust_n`、参考高度和竖直速度；不能把问题直接转成奖励调参。
原有 Direct-GPU reset 报错并未通过隐藏日志处理；reset 深度/位姿回归仍必须通过。

## 3. 用旧模型对照，不重训旧模型

```bash
python -u training/scripts/eval_isaac_single_ppo.py \
  runs/isaac_nav_curriculum/open/checkpoint_final.pt \
  --scenario open_target --no-headless --visualize --steps 600 \
  --trace runs/timing_v2/old_actor.trace.jsonl \
  --output runs/timing_v2/old_actor_eval.json
```

自动识别为 `hybrid_v1`。修正后的步进可能改变轨迹，因此是修复前后对照，不承诺与旧错误轨迹完全一致。
当前固定目标课程会屏蔽旧候选动作；通用 smoke 旧策略评估继续保留历史合同。
先确认不再起飞后迅速下坠。旧模型依然可能导航失败：物理修复不等于学会避障。

## 4. 开始新网络短训练与课程

仅在步骤 2 通过后运行，输出到新目录，**不要 resume 旧九维模型**：

```bash
python -u training/scripts/train_isaac_single_ppo.py \
  --scenario open_target --headless --total-steps 1024 \
  --output runs/isaac_navigation_v2/open
python -u training/scripts/eval_isaac_single_ppo.py \
  runs/isaac_navigation_v2/open/checkpoint_final.pt \
  --scenario open_target --headless --steps 600 \
  --trace runs/isaac_navigation_v2/open/short.trace.jsonl \
  --output runs/isaac_navigation_v2/open/short_eval.json
```

1024 步只检查链路，不是学会标准。续训可使用：

```bash
python -u training/scripts/train_isaac_single_ppo.py \
  --scenario open_target --headless --total-steps 20000 \
  --resume runs/isaac_navigation_v2/open/checkpoint_final.pt \
  --output runs/isaac_navigation_v2/open
python -u training/scripts/eval_isaac_single_ppo.py \
  runs/isaac_navigation_v2/open/checkpoint_final.pt \
  --scenario open_target --headless --steps 15000 \
  --output runs/isaac_navigation_v2/open/eval_50.json
```

`--total-steps` 是累计 transition 目标，不是追加步数。恢复 optimizer 会恢复其学习率，单改 YAML 的 lr
不会自动覆盖恢复值。本轮不调 lr 或控制器增益。

升级标准为完整回合 ≥50、`collision_free_success_rate≥0.90`、`out_of_bounds_episode_rate=0`。
报告区分 `run_completed`、`physics_timing.valid` 和 `curriculum_passed`。
open / wall_edge / wall_avoidance 分别可用最多 15000 / 20000 / 30000 控制步获取至少 50 回合。
达到上一关标准后，以同一 v2 checkpoint 续训下一关，并输出到独立目录。固定地图高成功率不证明随机地图泛化。

## 5. 规则目标接口与下一阶段

`rule_goals.py` 的候选输入必须携带由实际观测地图产生的 `known_free/inflated_safe/connected/in_task_region`
证据、地图路径长度与预估增益。筛选半径 8 m、膨胀至少 0.5 m；评分为
`gain / (path_length / 1 m/s + abs(wrapped_yaw_delta) / 1 rad/s + 1 s)`，同分按 ID。
当前目标锁定至显式完成/失效/停滞/超时/重分配事件；失败冷却 10 s；无候选返回 `None` 与 `waiting`，
不宣告探索完成。已完成候选 ID 本回合不重复选择；重新出现的新 frontier 由未来地图层分配新 ID。

当前 Isaac 的固定课程目标使用 `LocalGoal` 接口，但没有真实候选地图。
`queue_local_goal()` 在当前动作奖励结算后、下一次 observation 发布前切换目标，清理停滞历史；
不清空 GRU、不以新目标距离计算旧动作进展。未来地图层应在动作边界触发失效与重选，不可直接修改目标张量。
`None` 的等待/扫描/重分配执行状态尚需在真实探索 backend 中接入，不能把 `None` 当作有效目标交给固定课程 backend。

事件区分：`navigation_reached` 是位置到达，`navigation_pose_reached` 还包括朝向/倾斜；
`new_frame_processed` 仅指成功处理一次深度采集，不是地图积分凭证。
`observation_completed` 必须由新帧融合或有效 frontier 覆盖驱动，当前 Isaac 固定课程恒为 false。
`goal_reached`/成功率在当前课程中表示导航姿态达标，不表示探索完成。

后续接 0.5 m 深度占据地图与 frontier；只用已知连通性搜索估代价，不生成执行轨迹、不构造 ESDF。
保留 16 机、150×150×5 m、D455M 20 m 的最终目标，本轮不扩大仿真规模。

## 如何根据结果调整

| 现象 | 先检查 | 不应立即做的事 |
|---|---|---|
| GUI 下坠、headless 正常 | 实际步数、render 额外步、每步推力、reset 位姿 | 加碰撞奖励、重训 |
| 时序一致且悬停正常，但 PPO 仍下坠 | actor vz、参考 z、实际 vz、控制器跟踪 | 放宽高度边界 |
| 指令 ≤2 而实际速度 >2 | 推力/参考跟踪超调、制动、事件发生的物理子步 | 直接裁剪物理速度 |
| 目标附近徘徊 | 位置、yaw、倾斜、停滞轨迹及 KL 分项 | 放宽到达阈值凑成功 |
| 规则无候选 | 实测地图、膨胀层、任务区域、连通性和冷却 | 强制 candidate 0 有效 |
| 覆盖率为 0 且 unavailable | 当前仍为导航课程，不存在真实融合地图 | 把视锥绘图当已探索区域 |

日志同时保留名义控制时间和实际物理累计时间；当前图中的速度仍是控制步末采样，不能当作物理子步速度峰值。
