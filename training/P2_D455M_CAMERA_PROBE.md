# P2 第一步：D455M 固定机载深度相机探针

P1 已验证 Hummingbird、Lee 控制器、120 Hz 物理、20 Hz 控制、接触事件、reset
和正常退出。本探针只闭合下一条链路：

```text
base_link 固定外参 -> Isaac USD Camera -> Replicator metric depth
 -> 0.9~20 m 有效范围 -> 64x40 normalized inverse-depth
```

它尚不是 PPO 训练环境，也没有体素建图、frontier 或任务分配。

## 已实现的约束

- 相机 prim 是 `base_link` 的子节点，策略没有相机云台动作；改变视线只能转动机体。
- 原始图像为 640×360；同时读取 `distance_to_image_plane` 和
  `distance_to_camera`，建图以后使用前者按内参反投影。
- 近端 0.9 m、远端 20 m；无效、过近和过远像素在 actor 输入中为 0。
- inverse-depth 归一化公式与 smoke backend 相同。
- 640×360 到 64×40 使用 inverse-depth 自适应最大池化，优先保留细小近障碍，
  不用双线性插值冲淡树干和墙边缘。
- 探针先正向观察 `contact_wall`，再把机体偏航 90° 观察 `wall_north`；两次中心
  深度都由场景几何和安装前向偏移推导，用于检查固定外参和偏航跟随。
- 每次读取前渲染 8 帧，规避 Isaac Sim 2023.1 reset 后前几帧可能陈旧的问题。

当前安装位置 `[0.25, 0, 0] m` 和仓库中的内参仍是仿真假设，不是真机标定结果。
探针为了兼容性把 annotator 输出复制到 CPU；这适合单相机几何验收，不是最终 16 机
并行训练的数据通路。正式 backend 需要在确认 Isaac 2023.1 支持后优先保留 GPU tensor，
并实测渲染、复制和预处理耗时。

## Ubuntu 4090 上运行

同步下列新增/修改文件；不要覆盖 `training/third_party`，避免删除已经编译的扩展：

```text
training/configs/isaac_single.yaml
training/training/racer_rmappo/d455m_sensor.py
training/training/racer_rmappo/isaac_single_env.py
training/training/scripts/diagnose_isaac_backend.py
training/training/tests/test_isaac_single_probe.py
training/P2_D455M_CAMERA_PROBE.md
training/README_RMAPPO.md
```

先跑不依赖 Isaac 的回归：

```bash
cd /home/lww4090/racer_ws/src/RACER-main/training
python -m pytest -q training/tests/test_isaac_single_probe.py
```

再加载 Isaac 环境并只跑相机，便于定位 renderer 问题：

```bash
source "$ISAACSIM_PATH/setup_conda_env.sh"
PYTHONUNBUFFERED=1 python -u training/scripts/diagnose_isaac_backend.py \
  --probe camera \
  --headless \
  --output runs/isaac_single_probe/p2_d455m_camera.json
echo $?
```

通过标志应为：

```text
P1_PROBE_RESULT=PASS
P1_PROBE_ENV_CLOSED
```

且 `echo $?` 为 0。脚本输出中的 `P1_PROBE` 是既有诊断协议名，不表示相机仍属于
物理 P1。

重点检查 JSON：

- `probes.camera.resolution == [640, 360]`；
- `actor_depth_shape == [40, 64]`；
- 正向墙中心深度约为 `4.0 - 0.5 - 0.25 = 3.25 m`；
- 偏航后北墙中心深度约为 `6.0 - 0.4 - 0.25 = 5.35 m`；
- 两次绝对误差均不超过 0.30 m；
- `depth_within_clipping_range=true`，inverse-depth 在 `[0,1]`；
- `camera_prim_path` 位于 `.../Hummingbird_0/base_link/D455M`。
- 记录 `render_frames_per_second` 作为单相机 headless 基线；它不是 16 机可达到的帧率。

相机单项通过后，再回归全量，确认加入 renderer 没破坏 P1：

```bash
PYTHONUNBUFFERED=1 python -u training/scripts/diagnose_isaac_backend.py \
  --probe all \
  --headless \
  --output runs/isaac_single_probe/p2_full_regression.json
echo $?
```

## 结果驱动的排查

- 深度全无效：先检查 headless RTX 是否启动、render product 是否创建，再检查相机
  OpenGL 光轴是否为机体 +X；不要先放宽验收阈值。
- 正向深度正确、偏航深度错误：检查相机是否真的挂在 `base_link` 下，以及 reset/
  teleport 后是否完成了 8 帧 renderer 刷新。
- 两个深度都有固定偏差：核对安装位置、墙体半尺寸和 USD stage 单位是否为米。
- 偶发读到上一姿态画面：把 `warmup_render_frames` 增至 12 或 16，并记录额外渲染
  成本；不能把旧帧当成新观测融合。
- 相机探针通过但进程退出崩溃：确认日志先出现 `P1_PROBE_ENV_CLOSED`；随后检查
  annotator detach 与 Kit 版本，不应进入训练。

## 通过之后

下一步不是立刻长时间 PPO，而是按
[P2_SINGLE_NAV_BACKEND.md](P2_SINGLE_NAV_BACKEND.md) 运行单机固定目标 backend 合同探针，
验证真实深度、本机状态、动作、奖励、done 和 episode 指标。该闭环稳定后才开始单机
稀疏静态场景训练。
