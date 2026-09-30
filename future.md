> 当前实施路线（2026-09-30）：规则选局部目标，PPO 只学四维避障导航；训练、评估均在 Isaac Sim。操作见 [navigation_v2](training/NAVIGATION_V2.md)，阶段见 [实施计划](ISAAC_RMAPPO_实施计划.md)。下文为历史讨论，候选头学习、残差、选择 mask 和 ROS 优先路线不再适用于当前方案。

1、替换transformer

可以引入 Transformer。结合当前代码，我最建议先采用：

**深度 CNN + Transformer 信息融合 + GRU 时间记忆 + PPO 训练。**

优先改 actor 中的“邻机、候选点与自身状态融合”模块。至于是否比当前网络更好，需要在同一仿真环境下比较碰撞率、探索效率和推理延迟；单靠网络结构不能确定。

你图里的结构有两个地方需要说清楚：

- PPO 是训练网络参数的算法，部署时运行的是训练好的策略网络，并不会再经过一个“PPO 计算模块”。
- 当前低层动作是 `[vx, vy, vz, yaw_rate]`，最后一项是机体偏航角速度。候选观测点另有目标 yaw，相机随机体转动。

当前实现集中在 [model.py](/Users/yoloflps/Desktop/PROJECT/explore/RACER-main/training/training/racer_rmappo/model.py:43)，可改的位置如下：

| 模块 | 当前实现 | Transformer 可以做什么 | 建议 |
|---|---|---|---|
| actor 信息融合 | 邻机、候选点分别编码并最大池化，再与深度和状态拼接 | 学习目标、障碍、邻机之间的关系 | **优先尝试** |
| centralized critic | 各无人机编码后，用均值和最大值汇总 | 学习编队内无人机之间的相互影响 | 可以作为第二项实验，仅影响训练端 |
| 时间记忆 | GRU | 用历史观测之间的注意力代替递归记忆 | 后续尝试，涉及训练缓存和部署接口 |
| 深度编码 | 小型 CNN | 替换为视觉 Transformer | 当前 64×40 输入下，建议先保留 CNN |

当前候选选择已经使用了简单的 `query/key` 点积打分，但还没有用完整的 Transformer 联合处理候选、邻机和视觉特征。

你图里的方案更接近第一种：把不同对象变成 token，让它们相互交换信息。例如，“左侧候选信息增益高，但有邻机正在靠近；右侧候选增益稍低，但通路更空”。这种对象集合建模与 [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) 的思路相符，不过这只是适用性依据，还不能证明在本项目中一定提高性能。

建议第一版这样组织：

```text
深度帧 → CNN → 若干带空间位置信息的视觉 token
自身状态 ───────────────→ 状态 token
已选目标 ───────────────→ 目标 token
RACER 候选点 ──────────→ 最多 16 个候选 token
通信范围内邻机 ─────────→ 最多 5 个邻机 token
                              ↓
                    小型 Transformer 融合
                              ↓
                         GRU 时间记忆
                              ↓
                 速度动作 / 候选选择 / 目标残差

                    PPO 根据采样轨迹更新网络参数
```

初始规模可以用 `token_dim=128`、2 层、4 个注意力头，并先设 `dropout=0`，减少采样与 PPO 更新时额外随机性带来的干扰。这些是实验起点，不是已经验证的最优参数。

实现时要保留几项约束：

- 无效候选和失联邻机必须通过 mask 屏蔽；无邻机时也要正常输出。
- 候选选择仍只在 `decision_mask=1` 时生效，速度动作继续按 20 Hz 输出。
- 邻机 token 使用相对位置、速度、消息年龄等信息，避免依赖固定无人机编号或输入排列。
- actor 只能读取本机可获得的信息；训练期 critic 的全局状态不能混入 actor。

另外，**Transformer 融合多个对象，与 Transformer 处理时间序列，是两件事。** 如果输入的 token 都来自当前时刻，它只学习当前对象之间的关系，不会自动记住几秒前的障碍。因此第一版保留 GRU 很合适。若以后改成时间 Transformer，还需加入历史缓存、时间编码、因果掩码和回合重置，并保证 PPO 更新时使用一致的历史上下文。普通 Transformer 在强化学习中的优化稳定性也需要注意，[GTrXL 论文](https://proceedings.mlr.press/v119/parisotto20a.html) 专门研究了这个问题。

图里的“局部地图特征、任务区信息”目前也没有完整进入 actor：现在主要输入深度特征、候选点和已选目标。如果加入局部地图，可以从本机占据地图提取 `unknown/free/occupied` 局部块，再压缩成 token；这不需要 ESDF。任务区边界等信息则需要同时补进训练环境和 ROS 推理输入。

验证效果时，先比较“现有 CNN+GRU”与“CNN+Transformer+GRU”，保持奖励、环境和训练步数一致，使用多个随机种子和未参与训练的场景。主要看无碰撞成功率、达到覆盖率所需时间、重复探索和机载推理延迟。当前更影响整体进度的仍是 Isaac backend 与部署链路尚未闭合。

---

`smoke` 是“冒烟测试环境”，意思是先用一个轻量环境检查训练程序能否完整运行。实现见 [smoke_env.py](/Users/yoloflps/Desktop/PROJECT/explore/RACER-main/training/training/racer_rmappo/smoke_env.py:1)。

它会真实运行：

```text
生成观测 → 策略输出动作 → 简化环境更新
        → 计算奖励 → PPO 反向传播 → 保存模型
```

所以它可以检查网络维度、候选 mask、动作限幅、GRU 重置、奖励计算、PPO 更新和 checkpoint 保存。以后加入 Transformer，也应先用 smoke 检查这些接口。

但它的场景是简化圆柱障碍，飞行是运动学更新，深度是近似生成的；候选点和候选增益也是合成的。它没有实际运行 RACER 的分区协商，也没有真实 UDP chunk 传输。因此：

- smoke 跑通，说明训练程序基本连通。
- smoke 的奖励上升，不能证明真实避障或协同探索能力提高。
- smoke 的覆盖率目前以整个地图盒为分母，只用于检查计算过程。
- 它默认没有 Isaac 图形界面，主要输出终端日志、指标文件和模型文件。

你可以直接在 CPU 上使用它，无须先安装 Isaac。当前终端的 Python 是 3.9.6，缺少 `torch`、`PyYAML` 和 `pytest`。如果已经安装 Conda，建议单独建环境：

```bash
conda create -n racer-smoke python=3.11 -y
conda activate racer-smoke
python -m pip install torch numpy pyyaml pytest
```

只运行 smoke 时，不需要执行会安装整套 Isaac/Orbit 依赖的 `training/setup.sh`。

先检查配置和单元测试：

```bash
cd /Users/yoloflps/Desktop/PROJECT/explore/RACER-main/training

python training/scripts/check_ros_training_config.py
python -m pytest -q training/tests/test_racer_rmappo.py
```

再运行一次短训练：

```bash
python training/scripts/train_rmappo.py \
  --backend smoke \
  --device cpu \
  --stage single_agent_sparse_static \
  --num-envs 2 \
  --total-steps 4096 \
  --output /tmp/racer_rmappo_smoke
```

这里 `--num-envs 2` 表示两个并行编队；当前课程每个编队只有一架无人机。`--total-steps` 按所有无人机累计的采样步数计算。完成后主要产生：

```text
/tmp/racer_rmappo_smoke/metrics.jsonl
/tmp/racer_rmappo_smoke/checkpoint_final.pt
```

随后评估：

```bash
python training/scripts/eval_rmappo.py \
  /tmp/racer_rmappo_smoke/checkpoint_final.pt \
  --device cpu \
  --stage single_agent_sparse_static \
  --num-envs 2 \
  --steps 1024
```

选择 1024 步是因为 smoke 默认回合上限为 512 步；评估过短可能没有完整回合，导致回合指标缺乏样本。查看结果时确认没有 `NaN/Inf`、checkpoint 能正常加载，并同时检查 `episode_count`。如果 `coverage_target_episode_count=0`，覆盖耗时显示 0 表示没有达标样本。

本轮核对了代码并给出结构建议，没有替换网络或安装依赖。






2、后续计划（当前路线）

执行依据：[ISAAC_RMAPPO_实施计划.md](ISAAC_RMAPPO_实施计划.md)。

1. 固定 Isaac/OmniDrones 运行环境与训练接口，不依赖 ROS。
2. 实现单机 Isaac backend：真实动力学、固定深度相机、速度飞控、碰撞与 reset。
3. 完成单目标避障与观测完成训练，再接入 0.5 m 深度占据建图。
4. 实现 RACER 启发的 frontier、区域任务与候选观测点；联合训练选点和导航。
5. 从两机开始接入独立地图、仿真 map chunks 通信与两两协商。
6. 逐步扩展至 4/8/16 机和 150×150×5 m，在 Isaac 内完成冻结模型评估。
7. 基线稳定后再比较 Transformer；ROS/真实 UDP/真机作为可选未来工作。

下方保留旧讨论供追溯，其中 ROS 验证前置、ROS actor 必须实现等安排不再适用于当前目标；完成度请以新计划的代码核对表为准。

<details>
<summary>历史方案：ROS 集成路线（已被 Isaac 内闭环计划替代）</summary>

你的项目目标是：

让 16 架无人机在 150 m × 150 m × 5 m 的未知环境中，使用固定机载 D455M 深度相机完成去中心化协同探索。

整体分工是：

```text
D455M 深度图
    ↓
每机 0.5 m 占据地图 + 在线 map chunks 融合
    ↓
RACER：frontier、HGrid 分区、两两协商、任务分配
    ↓
每机最多 16 个候选观测点
    ↓
共享 RMAPPO：
  1. 选择具体候选观测点
  2. 输出 vx、vy、vz、yaw_rate
  3. 学习避障并到达观测点
    ↓
位置和相机朝向正确、融合新深度帧
    ↓
继续生成新任务，直到可探索区域覆盖完成
```

RACER 保留高层任务分配和地图融合；RMAPPO 取代原来的局部观测点决策及 A*/B-spline 低层导航。训练时使用集中式 critic，部署时每架无人机只运行共享 actor。系统边界已经写在 [改造说明](/Users/yoloflps/Desktop/PROJECT/explore/RACER-main/RACER_RMAPPO_D455M_改造说明.md:5)。

当前项目已经有：

- RACER 候选任务消息与 PPO 选择接口；
- 占据地图、0.5 m 膨胀层和在线 chunks 融合；
- 速度动作桥接与 2 m/s 限幅；
- 共享 GRU actor、集中式 critic 和 RMAPPO 更新；
- 奖励、回合指标和 smoke 环境；
- 训练模型导出框架。

目前真正缺失的是：

- 高保真多机 Isaac backend；
- ROS 在线 actor 推理节点；
- Ubuntu/ROS 上的实际编译和 topic 联调；
- 最终训练模型。

证据很直接：[isaac_adapter.py](/Users/yoloflps/Desktop/PROJECT/explore/RACER-main/training/training/racer_rmappo/isaac_adapter.py:41) 目前只有接口定义，而 [trainer.py](/Users/yoloflps/Desktop/PROJECT/explore/RACER-main/training/training/racer_rmappo/trainer.py:38) 在选择 `isaac` 时会主动报错。`future.md` 当前还是空文件。

你下一步应按下面顺序推进。

1. 先在 Ubuntu ROS 机器验证已有代码

当前工作区有大量未提交修改，建议先保存一次 Git 快照。然后在 ROS 工作站执行：

```bash
cd /path/to/catkin_ws
catkin_make --pkg plan_env active_perception exploration_manager
source devel/setup.bash
```

先验证原 RACER，再验证 RL launch：

```bash
roslaunch exploration_manager single_drone_rl_d455m.xml \
  drone_num:=1 simulation:=true
```

需要确认：

- D455M 深度 topic 有数据；
- 0.5 m 地图正常更新；
- `RLTask` 能产生候选点；
- 没有 actor 时无人机保持悬停；
- 测试 selector 返回候选后，`RLTarget.active=true`；
- 速度桥的坐标转换、限幅和超时正常。

这是当前代码能否继续扩展的基础，验收顺序也记录在 [改造说明](/Users/yoloflps/Desktop/PROJECT/explore/RACER-main/RACER_RMAPPO_D455M_改造说明.md:305)。

2. 实现第一版单机 Isaac backend

这是最重要的下一个编码任务。先做 `single_agent_sparse_static`，不要直接上 16 机。



- 30 m × 30 m × 5 m；
- 一架现有无人机模型；
- 固定机载 D455M 深度相机；
- 直墙、L 墙、U 墙、圆柱树干；
- 固定或随机局部目标；
- 输出 `[vx, vy, vz, yaw_rate]`；
- 碰撞、越界、目标进展和停滞奖励；
- 完整 reset；
- 满足现有 `MultiUAVBackend` 张量合同。

该阶段可以把候选集合设成“一个真实目标 + 15 个 padding”，`decision_mask` 只在生成新目标时触发。先让 PPO学会看到深度后绕障到达目标。

第一阶段通过标准：

- 固定障碍场景能够过拟合；
- 无 NaN/Inf；
- 深度随机体 yaw 正确转动；
- 速度模长不超过 2 m/s；
- 碰撞率开始下降；
- 目标到达率明显上升；
- reset 后地图、目标、无人机和 GRU 状态全部清空。

3. 再加入 0.5 m 体素观测与探索奖励

单机避障跑通后，再实现：

- 深度视锥反投影；
- 本机新观测体素；
- 团队唯一体素；
- 重复观测；
- 可观测、可到达体素 mask；
- 50%、75%、90%、98% 覆盖率事件。

这里 actor 只能看到深度和本机可用信息；仿真真值只能用于奖励、碰撞检测和 coverage 分母。

4. 实现 ROS 在线 actor 节点

这个节点需要：

- 加载导出的 TorchScript；
- 订阅本机深度、里程计、`RLTask`、`RLTarget` 和邻机状态；
- 严格复用训练期 inverse-depth、64×40 resize 和三帧堆叠；
- 保存每架无人机的 GRU hidden state；
- 收到新 `RLTask` 时发布一次 `RLViewpointSelection`；
- 按 20 Hz 发布 `cmd_vel_body`；
- task 失效、模型异常或输入超时时清零速度并重置 hidden state。

即使没有训练好的模型，也可以先用随机或 smoke checkpoint 验证输入预处理、消息频率和安全悬停。

5. 然后接入 RACER 候选任务

这里存在一个需要特别处理的工程问题：不能为几十个并行 Isaac 编队各启动一套 16 机 ROS RACER 节点，速度和内存都难以承受。

建议采用两级方式：

- 大规模 Isaac 训练：使用与 `RLTask` 相同格式的轻量候选任务适配器，或者回放真实 RACER 任务数据；
- 单编队集成测试和后期微调：运行真正的 C++ RACER、ROS 地图融合和两两协商。

低层避障可以先用随机局部目标训练，不依赖 RACER。候选选择头随后再使用 RACER 候选数据训练，这与当前文档中的课程顺序一致：[README_RMAPPO.md](/Users/yoloflps/Desktop/PROJECT/explore/RACER-main/training/README_RMAPPO.md:67)。

6. 最后逐步扩展到多机

按以下顺序：

```text
1 机 / 30 m
→ 4 机 / 60 m
→ 8 机 / 100 m + 通信延迟和丢包
→ 16 机 / 150 m
→ ROS RACER 完整闭环
→ 真机低速测试
```

多机阶段再引入邻机状态、机间碰撞、团队唯一体素奖励和通信扰动。Transformer 暂时不应成为下一步；先以现有 CNN+GRU 建立可运行基线，之后再比较 Transformer 是否真正改善探索效率。

所以现在最具体的里程碑是：

> 在 Ubuntu + NVIDIA 训练机上，实现并跑通“单机 D455M 深度 + Isaac 动力学 + 随机局部目标 + RMAPPO 避障”的 `IsaacRMAPPOBackend`。

完成这个里程碑后，项目才真正进入可训练状态。目前 smoke 只证明 PPO 代码路径能运行，不能替代这一步。

</details>
