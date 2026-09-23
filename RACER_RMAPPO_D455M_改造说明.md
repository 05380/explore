# RACER + 共享 RMAPPO + D455M 改造说明

## 1. 结论与系统边界

本次采用的结构是：

```text
D455M 深度图
    -> 每机 0.5 m 占据/ESDF 地图
    -> 在线 map chunks 交换与融合
    -> RACER frontier/HGrid 分区和两两协商
    -> 每机 RLTarget 局部子目标
    -> 共享 RMAPPO actor（分散执行）
    -> 机体系速度动作
    -> PositionCommand / 飞控
```

RACER 继续负责：未知空间转 frontier、层次栅格任务分区、全局/局部视点选择、无人机间两两协商。PPO 不重新学习任务分配，只学习从当前深度观测到 RACER 子目标之间的避障与局部导航。

地图融合应在探索过程中持续进行，而不是等所有无人机结束后再做。在线融合后，各机能尽快利用队友的新观测更新 frontier 和任务分配；结束阶段只需要做完整度、冲突和覆盖率检查。代码中的 map chunks 已改为在线定时发送、接收后立即写入本机地图并触发膨胀和 ESDF 更新。

## 2. 本次已经实现的代码

### 2.1 RACER 只输出子目标的 RL 模式

新增参数 `fsm/use_rl_navigation`，默认 `false`，因此原有 RACER A*/kinodynamic/B-spline 行为保持不变。

打开该参数时：

- RACER 仍运行 frontier、HGrid、TSP 和两两协商；
- `planExploreMotion()` 计算 `next_pos`、`next_yaw` 后不再生成低层轨迹；
- FSM 发布 `exploration_manager/RLTarget`；
- RL 模式不发布 B-spline，也不调用原轨迹碰撞检查；
- FSM 使用真实里程计判断到达子目标，并按 2 s 默认周期重新评估 frontier/分区；
- DroneState 始终广播真实里程计状态，而不是预测的 B-spline 状态。

`RLTarget.msg` 包含：目标序号、无人机编号、世界系目标位置、目标 yaw、当前分配的 grid IDs，以及是否为返航目标。目标以 latched topic 发布并以 5 Hz 重发，策略晚启动时也能拿到目标。

每架无人机的目标 topic：

```text
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

范围应为 `[-1, 1]`。默认物理上限为前进 1.5 m/s、后退 0.5 m/s、横向 0.8 m/s、垂向 0.5 m/s、yaw 1.0 rad/s。桥接节点只做坐标变换、限幅、飞行边界限制和 0.3 s 命令超时悬停，不查询地图，也不替策略规划避障路径。

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
- 障碍膨胀：0.75 m；
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
- 远端 chunk 写入后更新障碍膨胀与 ESDF；
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

重要：新增 launch 不包含 PPO actor。actor 未启动时，动作桥接会因超时持续发悬停命令，所以无人机不会自己探索。这是预期的安全行为。

16 机 launch 是 ROS topic、分配与融合的集成测试入口，不是高保真训练器。`poscmd_2_odom` 是轻量运动学模拟，不能验证碰撞动力学、深度噪声或 sim-to-real。正式训练应由 Isaac Sim 发布每机深度、相机位姿和里程计，并订阅每机动作。

训练/部署参数合同在：

```text
swarm_exploration/exploration_manager/config/rmappo_d455m_16.yaml
```

## 4. Isaac Sim / navrl-training 应如何继续实现

当前工作区完成的是 RACER–RL 接口、地图参数和在线融合，不包含已经训练好的模型，也没有把 Isaac Sim 环境复制进 RACER。建议在 `navrl-training` 中按以下顺序改造。

### 4.1 先做单机局部导航任务

复用 navrl-training 的无人机动力学、PPO 网络、墙体采样、动作延迟和噪声随机化，但把原有 ray-cast lidar 观测替换或增加为 D455M 针孔深度图。actor 输入建议：

- 连续 3 帧 64×40 inverse-depth；
- 机体系速度、roll/pitch、`sin(yaw)`/`cos(yaw)`；
- RACER 子目标在机体系下的相对位置和目标 yaw；
- 上一时刻动作；
- 最近 5 架邻机的相对位置、相对速度、消息年龄和 valid mask。

先在 30×30×5 m、单机、稀疏静态障碍中确认能绕过直墙、L 墙和 U 墙，再加树木、楼房和密集窄通道。不要一开始就用 16 机 150 m 场景，否则碰撞与长期探索奖励会让 PPO 的信用分配非常困难。

### 4.2 再使用共享 RMAPPO

- 16 架无人机共享一个 actor 参数；
- 执行时每机只使用本机深度、本机状态、自己的 RACER target 和通信范围内邻机状态；
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
r = 目标距离进展
  + 本机新观测体素
  + 团队唯一新观测体素
  + 到达 RACER 子目标
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

## 7. 验收顺序

1. 先以 `use_rl_navigation=false` 回归原 RACER，确认旧 launch 不受影响。
2. 单机 RL launch，不启动 actor，确认 watchdog 悬停、D455M 深度和 0.5 m 地图正常。
3. 用固定脚本发布动作，验证 target、机体系/世界系转换、边界和超时。
4. 两机验证 HGrid 协商、UDPROS chunks、接收者过滤和断链补块。
5. 四机验证在线 union map 与重复体素奖励口径。
6. 在 Isaac Sim 完成单机 curriculum 后再接 4/8/16 机 RMAPPO。
7. 最后才上真机，先低速、软障碍、保护网和人工急停，再逐步提高速度与范围。

由于当前执行环境没有 ROS/catkin 工具链，本次只能完成代码静态检查和 launch XML 语法检查，尚未在本机完成 catkin 编译、ROS topic 联调或 Isaac Sim 训练。第一次在 ROS 工作站构建时，应完整清理对应包的旧消息生成缓存后重新 `catkin_make`。
