# RACER + RMAPPO 训练目录

2026-09-30：当前方案是“规则选择局部目标，PPO 只学避障到达”。先阅读
[navigation_v2 改造与验收](NAVIGATION_V2.md)，包含三模式下坠排查、旧模型评估和新模型训练命令。

本目录同时保留原始 `training/training/scripts/train.py` 单机 LiDAR PPO 基线，并新增
`train_rmappo.py`。两者不是同一个任务：旧脚本不能用来证明 16 机 D455M 协同策略已经训练完成。

## 当前完成度

新增训练链路已经实现：

- `[并行编队, agent, ...]` 张量合同；`num_parallel_swarms` 与无人机数量严格分离；
- 共享 D455M 深度 actor（3 帧 64×40 inverse-depth）和 GRU；
- 训练期集中式、对 agent 排列等变的 critic；
- navigation_v2：tanh Gaussian 四维速度 `[vx, vy, vz, yaw_rate]`，没有候选选择和目标残差；
- 规则选点模块使用已知地图的候选有效性、代价和增益输入；当前 Isaac 接固定课程目标，真实 frontier 仍待接；
- 递归序列 minibatch、GAE、PPO clip、value clip、梯度裁剪和 checkpoint；
- v2 保留目标进展、近障碍、机间距、动作变化、停滞、碰撞、越界与到达奖励，关闭新体素和候选先验奖励；
- smoke 内固定机载相机视锥的合成体素射线统计；相机没有独立动作，改变视线须转动机体；
- 区分位置到达、导航姿态达标、地图观测完成；当前 Isaac 没有地图融合，最后一项恒为 false；
- 历史混合策略的新体素计数/奖励封顶仅保留作兼容测试，不代表 v2 已接真实建图；
- 按完整回合统计成功、碰撞、障碍碰撞、机间碰撞、超时、停滞、越界、安全接管和达到覆盖率所需时间；
- 无 Isaac 依赖的 `smoke` 合同环境、评估、TorchScript actor 导出和单元测试。

`smoke` 环境只验证训练代码、维度、速度限幅、奖励与循环能否工作。它现在会把圆柱深度图沿固定机载相机视锥反投影并统计射线经过的体素，不再把无人机所在体素当成“新观测”；但它仍没有真实飞行动力学、纹理、完整建筑网格或真实传感器噪声，产出的权重不得用于真机，也不代表避障训练完成。

高保真 `isaac` backend 目前完成的是单机导航子集，不是最终多机探索 backend。P1 已验证 Hummingbird、Lee 控制器、2 m/s 指令限速、接触力、越界和 reset，见 [P1_ISAAC_SINGLE_PROBE.md](P1_ISAAC_SINGLE_PROBE.md)。P2 又验证了固定机载 D455M 的 640×360 metric depth、0.9～20 m 范围、机体偏航跟随、64×40 inverse-depth、完整 PPO 观测/奖励合同和连续 20 回合 reset 生命周期，见 [P2_D455M_CAMERA_PROBE.md](P2_D455M_CAMERA_PROBE.md)、[P2_SINGLE_NAV_BACKEND.md](P2_SINGLE_NAV_BACKEND.md) 和 [P2_SINGLE_NAV_LIFECYCLE.md](P2_SINGLE_NAV_LIFECYCLE.md)。旧 `env.py` 是单机 4 m LiDAR 任务，不用于本项目。通用 `train_rmappo.py --backend isaac` 仍主动拒绝没有生命周期所有者的调用；单机 Isaac 必须使用专用入口。backend 遵守
`training/training/racer_rmappo/isaac_adapter.py` 的 v2 合同返回深度、ego、已选 target、邻机状态和 centralized critic state；旧候选字段仅在 hybrid_v1 评估存在。

P3 已增加墙后目标、确定性绕墙可达性探针、单机 Isaac PPO 训练入口和确定性评估入口，见 [P3_SINGLE_WALL_PPO.md](P3_SINGLE_WALL_PPO.md)。它只训练四维导航头；规则选择器已有独立实现，但真实地图、frontier 候选生成、多机通信和任务协商仍未接入，因此不能称为完整探索系统，也不能直接部署。之前 P1/P2 的通过记录不替代本次 GUI/headless 三模式回归。

## 环境诊断

训练机应为 Ubuntu x86_64、NVIDIA GPU，并与仓库内旧版 Isaac Sim/Orbit/OmniDrones 依赖匹配。先运行：

```bash
cd /path/to/RACER-main/training
python training/scripts/diagnose_rmappo.py
# 只有验证历史 ROS 路线时才需要 check_ros_training_config.py。
```

当前 macOS/Apple Silicon 工作站没有 CUDA，只适合静态检查；即使安装 CPU PyTorch，也只能运行小规模 smoke 测试。

`check_ros_training_config.py` 仅用于历史 ROS 接口，比较 YAML 与 ROS launch 的地图、相机、速度及旧混合动作等参数；不作为 Isaac-only 训练前置条件。

## 先跑合同测试

在已经执行 `setup.sh` 的训练环境中：

```bash
cd /path/to/RACER-main/training
pytest -q training/tests/test_racer_rmappo.py
python training/scripts/train_rmappo.py \
  --backend smoke --device cpu --stage single_agent_sparse_static \
  --num-envs 2 --total-steps 4096 --output /tmp/racer_rmappo_smoke
python training/scripts/eval_rmappo.py \
  /tmp/racer_rmappo_smoke/checkpoint_final.pt \
  --device cpu --num-envs 2 --steps 256
```

确认 loss、KL、entropy 都为有限值，reward 会变化，并且测试通过后，才开始实现/运行 Isaac backend。

回合指标以 `episode/count` 为分母。`env/collision_step_rate` 只是发生碰撞的仿真步比例，不能当成回合碰撞率；验收主要查看 `episode/collision_rate`、`episode/collision_free_success_rate`、两类碰撞率、超时率和 `episode/time_to_coverage_s_mean`。没有完整回合结束时回合率暂记为 0，同时 `episode/count=0`，不能把这个 0 解读成安全。

覆盖率必须定义为“团队已观测体素数 / 本回合可观测且可探索体素数”。Isaac 场景生成时应保存 ground-truth 可探索 mask，排除建筑实体内部、地图外部和永久不可达空间；否则 98% 可能在数学上不可达。`smoke` 环境为了检查张量合同暂以整个长方体为分母，而且无障碍方向没有合成深度返回，因此不能用它的覆盖率或 `time_to_coverage` 判断探索性能。评估时必须同时查看 `coverage_target_episode_count`，该值为 0 时，均值 0 秒表示“无样本”，不是瞬间完成。

新策略应确认动作 `[E,A,4]`，观测不含 candidates/decision_mask。旧候选 mask 测试仅用于 hybrid_v1 兼容评估。

## 课程顺序

配置真源为 `swarm_exploration/exploration_manager/config/rmappo_d455m_16.yaml`。建议逐阶段训练并使用前一阶段 checkpoint 初始化：

1. `single_agent_sparse_static`：1 机，30×30×5 m；
2. `single_agent_open_target`：1 机，空旷近目标，学习跟踪、制动与 yaw；
3. `single_agent_wall_edge`：1 机，沿墙边缘安全通过；
4. `single_agent_wall_avoidance`：1 机，固定墙后目标，作为完整绕障验收；
5. `four_agent_dense_static`：4 机，60×60×5 m；
6. `eight_agent_comm_randomization`：8 机，100×100×5 m；
7. `sixteen_agent_full`：16 机，150×150×5 m。

每个规模均由规则选择目标、PPO 学导航，不再训练候选头或联合微调选点。当前先过三模式物理一致性，再进行固定近目标、墙边、绕墙课程，最后接真实建图与规则选点。

当前单机墙体阶段使用拥有完整 `SimulationApp` 生命周期的专用入口：

```bash
python training/scripts/train_isaac_single_ppo.py \
  --scenario open_target --headless --total-steps 1024 \
  --output runs/isaac_navigation_v2/open
```

当前入口严格限制为一个物理环境、一架无人机。后续 `num-envs` 表示并行编队数，不是无人机数；在实现批量物理场景和相机前不能只改参数宣称已经并行。

## 导出

```bash
python training/scripts/export_rmappo_actor.py \
  runs/stage4/checkpoint_final.pt deployment/rmappo_actor.ts
```

导出时同时生成 `.json` 元数据。任何推理入口都必须复用相同的 inverse-depth、resize、frame stack、字段顺序和动作缩放；当前仍在 Isaac 内训练评估，不新增 ROS 节点。

v2 导出输入顺序为 `depth,ego,target,neighbors,hidden`，输出四维归一化动作和新 hidden；元数据包含 policy_spec。旧 hybrid_v1 的七输入、九动作导出仅供历史评估，不作为 v2 的 ROS 部署合同。

相机是刚性安装，不存在独立云台动作。机体完整姿态决定相机位姿；规则提供目标 yaw，策略通过 `yaw_rate` 转动机体。当前 Isaac 导航姿态达标要求位置误差 ≤0.5 m、yaw 误差 ≤0.25 rad、倾斜 ≤0.20 rad。未来只有融合相应新深度帧或 frontier 有效覆盖，才能报告地图观测完成。

## 地图融合

以下为后续探索阶段的设计，不是当前 Isaac 导航 backend 的已实现功能。地图应在探索过程中通过 map chunks 增量融合，不需要结束后再执行一个独立的“融合任务”。结束时需要：

1. 停止产生新目标；
2. 等待 chunk stamp/缺块查询收敛；
3. 检查所有在线无人机的 world 原点、分辨率和地图边界一致；
4. 由指定节点保存团队 union map 和覆盖率统计。

若各机 VIO 坐标系没有预先对齐，必须先做外参或多机位姿图对齐；线性体素地址本身无法修正坐标漂移。UDPROS 丢包依靠缺块查询最终补齐，调试时要验证断链恢复后的 chunk index 是否真正收敛。

## 调试重点

- 首先检查 observation 每个字段的 `shape/min/max/nan`，尤其 inverse-depth 的无效值必须为 0；
- 接入真实候选后，记录有效候选数、规则分数、拒绝原因、锁定目标 ID 和冷却状态；v2 没有选点熵和目标残差；
- 分别检查平移速度指令模长 ≤2 m/s 与实际速度超调，不要把指令限幅当成实际速度硬约束；
- 分项画 reward 曲线。v2 探索相关项应为 0；近障碍项在近障场景仍为 0 时检查距离查询与坐标系；
- 近障安全距离会随速度增大：`1.0 + v*0.30 + v^2/(2*1.0)`，最大 3.5 m。若仿真实测最大减速度或端到端延迟更差，应先改这两个参数；
- 记录 `approx_kl`、`clip_fraction`、entropy、gradient norm。KL 突升通常先减学习率或 epoch；entropy 很快归零先检查奖励尺度；
- 分别统计障碍碰撞、机间碰撞、越界、停滞、到达 target 和覆盖率，不要只看总 reward；
- 以 1 机固定种子场景过拟合为第一项验收，再做随机场景；
- 16 机前先做 2/4 机通信断连、延迟、乱序和恢复；
- 真机必须保留飞控急停/保护圈。PPO 是主避障器不等于可以移除独立的最终安全保护。
