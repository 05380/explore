# RACER + RMAPPO 训练目录

本目录同时保留原始 `training/training/scripts/train.py` 单机 LiDAR PPO 基线，并新增
`train_rmappo.py`。两者不是同一个任务：旧脚本不能用来证明 16 机 D455M 协同策略已经训练完成。

## 当前完成度

新增训练链路已经实现：

- `[并行编队, agent, ...]` 张量合同；`num_parallel_swarms` 与无人机数量严格分离；
- 共享 D455M 深度 actor（3 帧 64×40 inverse-depth）和 GRU；
- 训练期集中式、对 agent 排列等变的 critic；
- tanh Gaussian 四维动作 `[vx, vy, vz, yaw_rate]`；
- 递归序列 minibatch、GAE、PPO clip、value clip、梯度裁剪和 checkpoint；
- 目标进展、新体素、团队唯一体素、重复观测、近障碍、机间距、动作变化、停滞、碰撞和越界的分项奖励；
- 新体素奖励单步封顶，防止一帧深度生成的大量体素压倒碰撞惩罚；
- 无 Isaac 依赖的 `smoke` 合同环境、评估、TorchScript actor 导出和单元测试。

`smoke` 环境只验证训练代码、维度、速度限幅、奖励与循环能否工作。它用圆柱近似障碍且没有真实飞行动力学，产出的权重不得用于真机，也不代表避障训练完成。

尚未完成的是高保真 `isaac` backend。旧 `env.py` 是单机 4 m LiDAR 任务，代码会在选择 `--backend isaac` 时明确报错，防止误训。Isaac backend 必须按
`training/training/racer_rmappo/isaac_adapter.py` 返回深度、ego、RACER target、邻机状态和 centralized critic state。

## 环境诊断

训练机应为 Ubuntu x86_64、NVIDIA GPU，并与仓库内旧版 Isaac Sim/Orbit/OmniDrones 依赖匹配。先运行：

```bash
cd /path/to/RACER-main/training
python training/scripts/diagnose_rmappo.py
```

当前 macOS/Apple Silicon 工作站没有 CUDA，只适合静态检查；即使安装 CPU PyTorch，也只能运行小规模 smoke 测试。

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

## 课程顺序

配置真源为 `swarm_exploration/exploration_manager/config/rmappo_d455m_16.yaml`。建议逐阶段训练并使用前一阶段 checkpoint 初始化：

1. `single_agent_sparse_static`：1 机，30×30×5 m；
2. `four_agent_dense_static`：4 机，60×60×5 m；
3. `eight_agent_comm_randomization`：8 机，100×100×5 m；
4. `sixteen_agent_full`：16 机，150×150×5 m。

示例命令（只有 Isaac backend 实现后才可用）：

```bash
python training/scripts/train_rmappo.py \
  --backend isaac --device cuda:0 --stage single_agent_sparse_static \
  --num-envs 64 --total-steps 20000000 --output runs/stage1

python training/scripts/train_rmappo.py \
  --backend isaac --device cuda:0 --stage four_agent_dense_static \
  --num-envs 16 --total-steps 40000000 --resume runs/stage1/checkpoint_final.pt \
  --output runs/stage2
```

`num-envs` 表示并行编队数，不是无人机数。可并行编队数必须从 2/4 开始逐步增加，以 GPU 显存和仿真 FPS 为准。

## 导出

```bash
python training/scripts/export_rmappo_actor.py \
  runs/stage4/checkpoint_final.pt deployment/rmappo_actor.ts
```

导出时同时生成 `.json` 元数据。ROS 推理节点必须严格复用相同的 inverse-depth、resize、frame stack、字段顺序和动作缩放；任何一项不一致都会造成 sim-to-real 输入漂移。

## 地图融合

地图在探索过程中通过 map chunks 增量融合，不需要结束后再执行一个独立的“融合任务”。结束时需要：

1. 停止产生新目标；
2. 等待 chunk stamp/缺块查询收敛；
3. 检查所有在线无人机的 world 原点、分辨率和地图边界一致；
4. 由指定节点保存团队 union map 和覆盖率统计。

若各机 VIO 坐标系没有预先对齐，必须先做外参或多机位姿图对齐；线性体素地址本身无法修正坐标漂移。UDPROS 丢包依靠缺块查询最终补齐，调试时要验证断链恢复后的 chunk index 是否真正收敛。

## 调试重点

- 首先检查 observation 每个字段的 `shape/min/max/nan`，尤其 inverse-depth 的无效值必须为 0；
- 检查训练动作经缩放后平移速度模长不超过 2 m/s，ROS bridge 也会再次硬限幅；
- 分项画 reward 曲线。新体素项长期顶到 cap，说明计数或权重过大；近障碍项总为 0，说明距离查询或坐标系有错；
- 近障安全距离会随速度增大：`1.0 + v*0.30 + v^2/(2*1.0)`，最大 3.5 m。若仿真实测最大减速度或端到端延迟更差，应先改这两个参数；
- 记录 `approx_kl`、`clip_fraction`、entropy、gradient norm。KL 突升通常先减学习率或 epoch；entropy 很快归零先检查奖励尺度；
- 分别统计障碍碰撞、机间碰撞、越界、停滞、到达 target 和覆盖率，不要只看总 reward；
- 以 1 机固定种子场景过拟合为第一项验收，再做随机场景；
- 16 机前先做 2/4 机通信断连、延迟、乱序和恢复；
- 真机必须保留飞控急停/保护圈。PPO 是主避障器不等于可以移除独立的最终安全保护。
