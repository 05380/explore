# RACER + 共享 RMAPPO + D455M 改造说明

## 1. 结论与系统边界

本次采用的结构是：

```text
D455M 深度图
    -> 每机 0.5 m 占据地图 + 0.5 m 障碍膨胀层
    -> 在线 map chunks 交换与融合
    -> RACER frontier/HGrid 分区、两两协商、候选观测点生成
    -> 每机 RLTask（最多 16 个候选观测点）
    -> 共享 RMAPPO 选择候选点/小残差，并连续控制避障
    -> 经安全校验得到 RLTarget
    -> 机体系速度动作
    -> PositionCommand / 飞控
```

RACER 继续负责：未知空间转 frontier、层次栅格任务分区、无人机间两两协商以及安全候选观测点生成。PPO 不重新学习任务分配，但负责在本机已分配任务内选择具体候选观测点（可附加很小的 xyz/yaw 残差），并学习到达该点的避障与局部导航。

地图融合应在探索过程中持续进行，而不是等所有无人机结束后再做。在线融合后，各机能尽快利用队友的新观测更新 frontier 和任务分配；结束阶段只需要做完整度、冲突和覆盖率检查。代码中的 map chunks 已改为在线定时发送、接收后立即写入本机地图并更新障碍膨胀层。RL 专用 launch 关闭 ESDF 的周期计算；原 RACER launch 仍默认开启 ESDF，以保持兼容。

这里关闭的是 `updateESDF3d()` 的定时计算，不是删除 `SDFMap` 类型及其距离缓冲区；后者与原项目大量接口耦合，直接移除风险很高。RACER 的 frontier、HGrid、候选可见性和目标合法性仍必须保留占据概率、unknown/free 分类与膨胀占据层，它们不属于 PPO 低层轨迹规划器。

## 2. 本次已经实现的代码

### 2.1 RACER 输出任务和候选点，PPO 完成最终选择

新增参数 `fsm/use_rl_navigation`，默认 `false`，因此原有 RACER A*/kinodynamic/B-spline 行为保持不变。

打开该参数时：

- RACER 仍运行 frontier、HGrid、TSP 和两两协商；
- `planExploreMotion()` 不再用局部 tour 启发式决定唯一 `next_pos`，而是从已分配 frontier 中按轮询方式保留最多 16 个候选点；
- FSM 发布 `exploration_manager/RLTask`，包含 task/grid/frontier ID、候选世界坐标、yaw、所属 frontier 和估计可见体素数；
- PPO 通过 `RLViewpointSelection` 返回候选索引以及有界残差；
- FSM 将 xy 残差从选择时机体系旋转到 world 系，再检查 task/drone ID、候选索引、残差、所属分配网格、地图边界、已知自由状态和膨胀占据状态，通过后才发布激活的 `RLTarget`；
- RL 模式不发布 B-spline，也不调用原轨迹碰撞检查；
- FSM 使用真实里程计判断到达子目标；选择响应超时为 2 s，已选目标执行超时为 20 s，到达、目标失效、任务重分配或执行超时才重选；
- DroneState 始终广播真实里程计状态，而不是预测的 B-spline 状态。

新任务产生时先发布 `RLTarget.active=false`，明确取消旧目标；选择通过后发布同一 task ID 的 `active=true` 目标。任务和目标均为 latched topic，并以 5 Hz 重发。

每架无人机的目标 topic：

```text
/rl_navigation/task_<drone_id>
/rl_navigation/viewpoint_selection_<drone_id>
/rl_navigation/target_<drone_id>
```

### 2.2 PPO 动作到飞控命令的桥接

新增 `rl_velocity_bridge`。策略向以下 topic 发布 `geometry_msgs/TwistStamped`：

```text
/rl_navigation/cmd_vel_body_<drone_id>
```

默认把四个动作解释为归一化机体系动作：

```text
linear.x  前后速度
linear.y  横向速度
linear.z  垂向速度
angular.z yaw 角速度
```

范围应为 `[-1, 1]`。默认物理上限为前进 2.0 m/s、后退 0.5 m/s、横向 0.8 m/s、垂向 0.5 m/s、yaw 1.0 rad/s，并对三维平移速度做 2.0 m/s 模长硬限幅。桥接节点只做目标激活门控、坐标变换、限幅、飞行边界限制和 0.3 s 命令超时悬停；`RLTarget.active=false` 时即使 actor 仍误发旧速度也会悬停。它不查询地图，也不替策略规划避障路径。

输出：

```text
/planning/pos_cmd_<drone_id>
```

注意：这里的 watchdog 和边界限幅不是 RACER 避障兜底。实际真机仍应保留飞控级失联、姿态、最低/最高高度和紧急制动保护，这些保护不应作为 PPO 获取正常避障能力的捷径。

### 2.3 D455M 与 0.5 m 地图配置

新增 D455M 仿真配置：

```text
uav_simulator/local_sensing/params/camera_d455m_640x360.yaml
```

当前仿真参数按 640×360、水平 58°、垂直 35°估算：`fx=577.296`、`fy=570.840`、`cx=319.5`、`cy=179.5`。这是仿真近似值，不应覆盖真机标定；真机必须读取相机自己的 `CameraInfo`/出厂标定值。

专用 launch 使用：

- 实测有效最大深度：20.0 m；
- 最小建图深度：0.9 m；
- 体素分辨率：0.5 m；
- 地图范围：150 m × 150 m × 5 m；
- 障碍膨胀：0.5 m（当前 0.5 m 体素下为 1 个体素步长）；
- 有效飞行高度：0.5–4.5 m；
- HGrid 边长：10 m。

深度处理还修复了两个问题：

- 投影点缓存原来固定为 640×480，较大图像可能越界；现在按实际图像尺寸动态扩容；
- 原来无效深度和超过量程的深度会被写成最大距离，形成假的“20 m 障碍墙”；现在直接丢弃无效/超量程像素。

### 2.4 在线 map chunks 融合

原项目已经有 chunks 和缺块查询，但存在几个影响在线融合的问题。本次修改包括：

- `>= chunk_size` 时立即形成完整 chunk，修复正好等于 200 个体素时不发送的问题；
- 不足 200 个体素的尾块在默认 0.25 s 后也会封包，不再无限等待；
- 新体素以及 FREE/OCCUPIED 分类发生变化的体素都会进入后续 chunk；
- 接收端只处理 `to_drone_id` 指向自己的消息；
- 检查无人机编号、chunk 编号和地址/占据数组长度；
- 远端 chunk 写入后更新障碍膨胀；仅当 `map_ros/enable_esdf=true` 时继续更新 ESDF；
- `ChunkStamps` 携带发送机位置，可按通信半径过滤；
- 新配置默认使用 UDPROS，chunk 大小约 1 KB，降低 IP 分片风险；
- map 数量与“最后一个编号是否是地面站”解耦，16 号机不再被错误当成地面站。

专用配置中的通信半径为 30 m。chunks 可以由中间无人机继续转发，因此暂时失联后重新接近仍能补齐缺块。

UDPROS 是 ROS 1 点到点 UDP 协商，不等同于无需 ROS master 的底层无线广播。真机跨多台计算机时仍需稳定的 ROS 网络/VPN，或者在这些 ROS send/recv topic 外再接一个专用 UDP multicast/unicast 网桥。网络层必须处理 MTU、重传/补块、身份验证和时钟问题；本项目的 chunk index/缺块查询负责应用层最终补齐。

## 3. 新增配置和启动方式

单机集成：

```bash
roslaunch exploration_manager single_drone_rl_d455m.xml \
  drone_num:=1 simulation:=true
```

16 机 ROS 轻量联调：

```bash
roslaunch exploration_manager swarm_exploration_rl_d455m_16.launch
```

然后用 RViz 的 2D Nav Goal 或向 `/move_base_simple/goal` 发布任意 PoseStamped 触发探索。该 goal 只充当启动信号，不是最终探索目标。

重要：新增 launch 不包含 PPO actor。actor 未启动时，RACER 会因 2 s 选择超时重复生成任务，动作桥接会因超时持续发悬停命令，所以无人机不会自己探索。这是预期的安全行为。

不加载模型时，可先验证任务握手：

```bash
python3 $(rospack find exploration_manager)/scripts/rl_task_test_selector.py _drone_id:=1
```

该脚本只选择估计可见体素数最大的候选点，不发布飞行速度，不能当作策略使用。

16 机 launch 是 ROS topic、分配与融合的集成测试入口，不是高保真训练器。`poscmd_2_odom` 是轻量运动学模拟，不能验证碰撞动力学、深度噪声或 sim-to-real。正式训练应由 Isaac Sim 发布每机深度、相机位姿和里程计，并订阅每机动作。

训练/部署参数合同在：

```text
swarm_exploration/exploration_manager/config/rmappo_d455m_16.yaml
```

## 4. Isaac Sim / RMAPPO 训练端

`training/training/racer_rmappo` 已新增共享 GRU actor、集中式 critic、递归 PPO 更新、奖励组合、合同测试环境、评估和 TorchScript 导出。详细运行方式见 `training/README_RMAPPO.md`。当前仍不包含训练好的模型；高保真 Isaac backend 尚未接入，合同环境只用于验证张量和算法循环，不能代替避障训练。

### 4.1 先做单机局部导航任务

复用 navrl-training 的无人机动力学、PPO 网络、墙体采样、动作延迟和噪声随机化，但把原有 ray-cast lidar 观测替换或增加为 D455M 针孔深度图。actor 输入建议：

- 连续 3 帧 64×40 inverse-depth；
- 机体系速度、roll/pitch、`sin(yaw)`/`cos(yaw)`；
- RACER 候选点集合、候选 mask、相对位置、yaw、visible gain，以及已选目标；
- 上一时刻动作；
- 最近 5 架邻机的相对位置、相对速度、消息年龄和 valid mask。

先在 30×30×5 m、单机、稀疏静态障碍中确认能绕过直墙、L 墙和 U 墙，再加树木、楼房和密集窄通道。不要一开始就用 16 机 150 m 场景，否则碰撞与长期探索奖励会让 PPO 的信用分配非常困难。

### 4.2 再使用共享 RMAPPO

- 16 架无人机共享一个 actor 参数；
- 执行时每机只使用本机深度、本机状态、自己的 RACER task/target 和通信范围内邻机状态；
- 训练时 centralized critic 可以看到所有无人机状态、任务分配摘要和全局覆盖率；
- actor 使用 GRU/LSTM，应对有限视锥、遮挡、通信丢包和部分可观测性；
- 训练中随机打乱 agent ID，防止策略把固定编号当成角色。

推荐课程：1 机 30 m → 4 机 60 m → 8 机 100 m 加通信扰动 → 16 机 150 m。每一阶段都先冻结 RACER 的分区规则，避免任务分配和局部策略同时变化。

### 4.3 障碍物场景设计

建筑物应包含长直墙、L/U 形凹区、门洞、相邻楼间通道和不同高度屋檐；树木至少使用“树干碰撞体 + 树冠遮挡/碰撞体”，随机化树干半径、间距和冠层高度。生成后必须做以下有效性检查：

- 初始点周围至少 2 m 无碰撞；
- 场景主要自由空间连通，不能让大面积 frontier 物理不可达；
- 通道宽度分布包含容易、临界和不可通行三类，但不可通行区域不能占主导；
- 建筑与树木不能穿过地面或超出 5 m 场景导致错误深度；
- 训练和测试使用不同随机种子、几何组合与材质。

### 4.4 奖励函数

配置文件给出了第一版权重。计算时应分项记录，不要只记录总 reward：

```text
r = 已选目标距离进展
  + 小权重候选 visible-gain 先验
  + 本机新观测体素
  + 团队唯一新观测体素
  + 到达 PPO 所选局部目标
  - 碰撞/越界
  - 近障碍风险
  - 机间距离风险
  - 重复观测
  - 动作突变和无效悬停
  - 时间成本
```

关键定义：

- “本机新体素”只统计由本机传感器在当前 step 首次确认的体素，不能把刚收到的队友 chunk 再奖励一次；
- “团队唯一新体素”按团队 union map 的首次更新时间计算，防止 16 倍重复奖励；
- 目标进展使用 `d(t-1)-d(t)`，而不是直接奖励离目标近，否则容易学会停在目标附近；
- 碰撞应为终止惩罚，近障碍惩罚提供碰撞前的稠密梯度；
- 近障碍距离在训练中可以读取仿真碰撞几何，但 actor 不能读取这个特权信息；
- 探索奖励必须设单步上限，否则一帧深度图产生的大量体素会压倒碰撞惩罚；
- 覆盖率里程碑是团队奖励，使用过强会导致 credit assignment 恶化，需逐阶段加入。

可以复用 navrl-training 的 goal progress、stall/escape、VO 风险、动作噪声和 wall-follow teacher 思路，但 teacher 数据只能作为行为克隆预热或辅助 loss，不能在部署时变成隐藏规划器。

## 5. 必须注意的问题

### D455M 近距离盲区

建图最小深度设为 0.9 m，意味着已经进入 0.9 m 内的细杆、树枝或侧后方障碍可能没有有效深度。0.5 m 体素也可能丢失细树枝。即使 PPO 在仿真中表现很好，真机仍建议使用保护圈、短距传感器或飞控级紧急制动，并限制速度，使 20 Hz 控制周期和总延迟下有足够制动距离。

### 20 m 并不等于所有像素都可靠

20 m 是用户实测最高有效值，不应把 20 m 内所有回波当成同等可靠。训练要随机化远距离 dropout/噪声；真机可按置信度、材质和光照进一步缩短有效量程。代码现在不会把无效像素伪造成 20 m 障碍，但无回波也不会自动清出一条 20 m 自由射线，这是偏保守的选择。

### 0.5 m 分辨率改变了所有“体素个数”阈值

原来的 `cluster_min=100`、`min_unknown=4000` 等参数不能原样沿用。本次提供的是初值，不是最终标定结果。必须记录每个场景的 frontier cluster 大小、每格 UNKNOWN/FREE 数量、无任务次数和覆盖率曲线，再重新选阈值。

### 内存和计算量

150×150×5 m、0.5 m 分辨率约为 90 万体素。当前 SDF 多个 double/char/short 缓冲合计约 44 MB/无人机，16 个 ROS 进程仅核心地图就约 0.7 GB，实际加上 ESDF、frontier、深度和可视化会明显更高。20 m 深度和 `show_all_map=true` 也会增加 CPU/网络负担。正式 16 机联调建议关闭不必要 RViz 全图显示、降低深度处理频率，并分别记录每进程 CPU、RSS 和 chunk 带宽。

### 坐标系一致性

当前 chunk 中传输的是线性体素地址，接收端按自己的同尺寸地图直接解释；因此所有无人机必须共享相同的 world 原点、分辨率、地图尺寸和可靠的相对定位。原代码中的跨局部坐标变换仍是注释状态。若每机使用独立 VIO 原点，必须先做多机位姿图/外参对齐，再融合 chunk，否则地图会整体错位。

### UDP 与丢包

UDP 丢包是正常事件。当前协议依靠周期 stamp 和缺块查询最终补发，但 ROS topic 本身不提供加密、认证和严格拥塞控制。测试必须覆盖 5%/10%/20% 丢包、0–250 ms 延迟、短时断链和网络分区恢复，并检查 chunk index 是否最终收敛。

### 终止条件

不能只用“当前没有 frontier”立即判定全队完成，因为通信分区或相机遮挡也会暂时造成无 frontier。建议最终条件为：团队 union map 未知可达体素低于阈值，并且所有无人机连续若干秒没有有效 frontier，且 chunk stamp 已收敛。当前 FSM 保留了原项目的 IDLE 后再检查/返航逻辑，后续应在 Isaac/实机日志上标定等待时间。

## 6. 本次修改的文件

- `swarm_exploration/exploration_manager/msg/RLTarget.msg`
- `swarm_exploration/exploration_manager/msg/RLTask.msg`
- `swarm_exploration/exploration_manager/msg/RLViewpointSelection.msg`
- `swarm_exploration/exploration_manager/scripts/rl_task_test_selector.py`
- `swarm_exploration/active_perception/include/active_perception/hgrid.h`
- `swarm_exploration/active_perception/src/hgrid.cpp`
- `swarm_exploration/exploration_manager/src/fast_exploration_fsm.cpp`
- `swarm_exploration/exploration_manager/src/fast_exploration_manager.cpp`
- `swarm_exploration/exploration_manager/src/rl_velocity_bridge.cpp`
- `swarm_exploration/exploration_manager/launch/single_drone_planner.xml`
- `swarm_exploration/exploration_manager/launch/single_drone_rl_d455m.xml`
- `swarm_exploration/exploration_manager/launch/swarm_exploration_rl_d455m_16.launch`
- `swarm_exploration/exploration_manager/config/rmappo_d455m_16.yaml`
- `swarm_exploration/plan_env/src/map_ros.cpp`
- `swarm_exploration/plan_env/src/sdf_map.cpp`
- `swarm_exploration/plan_env/src/multi_map_manager.cpp`
- `swarm_exploration/plan_env/msg/ChunkStamps.msg`
- `uav_simulator/local_sensing/params/camera_d455m_640x360.yaml`
- `training/training/racer_rmappo/`（共享 actor、集中式 critic、奖励、rollout 和环境合同）
- `training/training/scripts/train_rmappo.py`
- `training/training/scripts/eval_rmappo.py`
- `training/training/scripts/export_rmappo_actor.py`
- `training/training/scripts/diagnose_rmappo.py`
- `training/training/tests/test_racer_rmappo.py`
- `training/README_RMAPPO.md`

## 7. 验收顺序

1. 先以 `use_rl_navigation=false` 回归原 RACER，确认旧 launch 不受影响。
2. 单机 RL launch，不启动 actor，确认 watchdog 悬停、D455M 深度和 0.5 m 地图正常。
3. 用固定脚本发布动作，验证 target、机体系/世界系转换、边界和超时。
4. 两机验证 HGrid 协商、UDPROS chunks、接收者过滤和断链补块。
5. 四机验证在线 union map 与重复体素奖励口径。
6. 在 Isaac Sim 完成单机 curriculum 后再接 4/8/16 机 RMAPPO。
7. 最后才上真机，先低速、软障碍、保护网和人工急停，再逐步提高速度与范围。

由于当前执行环境没有 ROS/catkin 工具链，本次只能完成代码静态检查和 launch XML 语法检查，尚未在本机完成 catkin 编译、ROS topic 联调或 Isaac Sim 训练。第一次在 ROS 工作站构建时，应完整清理对应包的旧消息生成缓存后重新 `catkin_make`。

## 8. 当前关键参数与训练端状态（续改）

当前运行/部署参数：

- 编队规模 16，最终空间 150×150×5 m，飞行高度 0.5–4.5 m；
- D455M 深度 640×360，约 58°×35°，有效范围 0.9–20 m；actor 输入为 3 帧 64×40 inverse-depth；
- 体素 0.5 m，障碍膨胀 0.5 m；深度建图每 4 像素抽样一次；
- HGrid 10 m，通信半径 30 m，chunk 200 体素，尾块 0.25 s 刷新，默认 UDPROS；
- 控制 20 Hz，动作超时 0.30 s，平移合速度硬上限 2.0 m/s；前向 2.0、后向 0.5、横向 0.8、垂向 0.5 m/s，yaw 1.0 rad/s；
- 每个 RACER task 最多 16 个候选点；选择响应超时 2 s，已选目标执行超时 20 s，1 m 内视为到达；位置残差限幅为 xy 各 1.0 m、z 0.5 m，yaw 0.35 rad；
- RMAPPO rollout 256，递归序列 32，PPO epoch 5，minibatch 8，`gamma=0.99`，`GAE=0.95`，clip 0.2，学习率 3e-4，GRU hidden 256；
- 新体素奖励分别按本机/团队唯一体素计数，但单步各封顶 1.0，重复观测惩罚单步最多 0.25；
- 近障碍奖励距离从低速 1.0 m 随速度按反应距离和制动距离增大，在 2 m/s 时封顶 3.5 m。

`training/training/racer_rmappo` 现在的共享 actor 输出 9 维混合动作：4 维连续速度、1 个离散候选索引、4 维有界 xyz/yaw 残差。候选选择的 log-prob/entropy 只在 `decision_mask=1` 的新任务事件参与 PPO loss；低层速度仍以 20 Hz 更新。合同冒烟环境可用于发现候选 mask、维度、递归状态、PPO 更新和奖励数值错误，但它不是飞行动力学训练环境。高保真 Isaac backend、真实 D455 深度渲染、楼房/树木碰撞体以及 ROS actor 推理节点仍需在 Ubuntu + NVIDIA 训练机上继续接入。不要把 smoke checkpoint 部署到无人机。

地图融合发生在探索过程中。结束时只需停止新任务、等待 chunk 缺块补齐和 stamp 收敛，然后保存 union map；如果各机 world/VIO 坐标未对齐，则必须先做坐标对齐，不能靠 chunk 合并消除重影。

训练、诊断、评估和导出命令以及逐项排错方法见 `training/README_RMAPPO.md`。

## 9. 改造后可能出现的问题与测试方法

| 风险 | 典型表现 | 如何确认 | 通过标准 |
|---|---|---|---|
| 消息未重新生成或 topic 接错 | 收不到 `RLTask`，或 target 永远 inactive | 清理旧 build/devel 后编译；`rostopic echo /rl_navigation/task_1`，再运行测试 selector | 同一 task ID 依次出现 task、selection、active target |
| 候选维度/掩码错误 | PPO 选到 padding，Categorical 出 NaN | 单测检查 `[env,agent,16,9]`；记录每批 valid 数和 logits | 每个 task 至少 1 个 valid；无 NaN/Inf；invalid 概率为 0 |
| 选择频率错误 | 每个 20 Hz step 都换目标、策略抖动 | 统计 `RLTask.task_id` 与控制 step；记录 `decision_mask` | 只在新任务、到达、失效、重分配、20 s 超时触发 |
| 残差导致非法目标 | FSM 持续打印 `Rejected PPO viewpoint` | 分别注入越界索引、超限残差、unknown/occupied 目标 | 非法选择不激活 target，合法零残差立即激活 |
| 关闭 ESDF 后误伤 RACER | frontier/HGrid 无结果或访问陈旧距离场 | `enable_esdf=false` 单机跑 frontier；对照 `true` 的 grid/frontier 数 | RL 模式持续产生任务；旧模式开启 ESDF 后行为不变 |
| 膨胀层或坐标系错 | 候选在墙内、融合地图重影 | RViz 同时显示 raw/inflated occupancy、候选点；检查 16 机地图参数 | 候选均 known-free 且 inflate=0；共享静态物体重合 |
| 20 m 深度造成 CPU/伪自由 | 回调积压、远处噪声清空障碍 | 记录深度回调周期、队列延迟、有效像素距离直方图 | 无效深度不变成 20 m 命中；控制与建图无持续积压 |
| UDP chunk 丢失/乱序 | 各机 coverage 长期不一致 | 注入 5/10/20% 丢包和 0–250 ms 延迟，断链后恢复 | 恢复后缺块补齐，chunk interval 和占据计数收敛 |
| 奖励投机 | 原地转圈刷体素、贴墙、重复扫 | 分项画 reward、团队唯一体素、重复体素、碰撞率 | 覆盖率增长伴随新区域访问，碰撞率与重复率不过阈值 |
| sim-to-real 输入漂移 | 仿真成功、真机静止或撞障碍 | 保存同场景预处理 tensor，逐元素比较训练/部署；检查坐标轴与单位 | inverse-depth、无效值、帧栈和 body/world 变换一致 |
| 16 机规模瓶颈 | ROS 队列堆积、任务过期、CPU/RSS 暴涨 | 按 1→2→4→8→16 机记录 callback latency、FPS、带宽、RSS | 99% 控制延迟低于 50 ms，选择低于 2 s，无持续丢队列 |
| 终止条件过早 | 通信分区后误判探索完成 | 暂时隔离两组无人机，再恢复通信 | frontier 空、chunk stamp 收敛、未知可达体素阈值同时满足才结束 |

建议严格按“消息接口 → 单机固定障碍 → 单机随机障碍 → 2/4 机通信 → 8/16 机规模 → 真机低速”的顺序验收。每一层未通过时不要用后续大场景掩盖问题。

## 10. 建议执行的测试命令

在 Ubuntu ROS 工作站上先重新生成消息并构建：

```bash
cd /path/to/catkin_ws
catkin_make --pkg plan_env active_perception exploration_manager
source devel/setup.bash
```

单机启动后检查参数与握手：

```bash
roslaunch exploration_manager single_drone_rl_d455m.xml drone_num:=1 simulation:=true
rosparam get /exploration_node_1/map_ros/enable_esdf
rostopic echo /rl_navigation/task_1
rosrun exploration_manager rl_task_test_selector.py _drone_id:=1
rostopic echo /rl_navigation/target_1
```

预期 `enable_esdf=false`；触发探索后 task 含 1–16 个等长候选字段；selector 返回选择后，同一 ID 的 target 从 `active:false` 变为 `active:true`。用 `_candidate_index:=999` 重启 selector 可验证越界选择被拒绝且无人机保持悬停。

训练端在装有 PyTorch/PyYAML 的环境运行：

```bash
cd /path/to/RACER-main/training
pytest -q training/tests/test_racer_rmappo.py
python training/scripts/train_rmappo.py \
  --backend smoke --device cpu --stage single_agent_sparse_static \
  --num-envs 2 --total-steps 4096 --output /tmp/racer_rmappo_smoke
```

smoke 只验收张量、混合动作、事件 mask、奖励和 PPO 更新，不验收真实避障。真正进入 Isaac Sim 前，必须实现 `MultiUAVBackend` 并让同一套测试在深度相机、碰撞体、动力学和通信扰动下通过。
