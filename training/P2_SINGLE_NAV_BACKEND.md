# P2 第二步：单机固定目标 Isaac backend 合同

本阶段把已验证的 Hummingbird 动力学和 D455M 深度接入现有 RMAPPO 张量合同，但不
执行 PPO 参数更新。目标是先证明下面的数据闭环真实可运行：

```text
规则提供固定目标
  -> depth/ego/target/candidate observation
  -> 归一化四维导航动作
  -> 物理速度指令与 Lee 控制器
  -> 新深度帧、状态、奖励、done 和 episode info
```

## 当前约束

- 只支持 `num_envs=1`、`num_agents=1`。
- 固定目标为 `[2.5, 0, 1.5] m`，位于正前方墙体之前，用于先验收数据合同。
- candidate 0 对应该固定目标，其余 15 个槽位无效。
- `decision_mask=0`，候选索引和残差分支不进入 PPO log-prob/entropy。
- actor 只使用 `[vx_body, vy_body, vz_world, yaw_rate]` 四维动作。
- actor 深度和近障奖励都使用已通过几何验收的 `distance_to_image_plane`；当前
  Isaac 2023.1 的 `distance_to_camera` 只作诊断，不能直接混入 clearance 奖励。
- 位置到达是 `navigation_reached`；只有位置、yaw、倾斜均合格且已读取新深度帧，
  才产生 `observation_completed/goal_reached`。
- 当前没有体素地图，因此 coverage 和 coverage-target 指标保持 0；固定目标完成不能
  冒充探索覆盖率成功。
- 回合因观测完成、碰撞、越界、停滞或超时结束，并自动 reset。

这仍不是正式训练 backend。`train_rmappo.py --backend isaac` 继续保持关闭，直到本合同
探针和固定场景 actor 连通性探针都通过。

## 需要同步的文件

```text
training/configs/isaac_single.yaml
training/training/racer_rmappo/d455m_sensor.py
training/training/racer_rmappo/isaac_single_env.py
training/training/racer_rmappo/isaac_single_backend.py
training/training/racer_rmappo/trainer.py
training/training/scripts/diagnose_isaac_backend.py
training/training/scripts/diagnose_isaac_navigation_backend.py
training/training/tests/test_isaac_single_probe.py
training/training/tests/test_isaac_single_backend.py
training/P2_D455M_CAMERA_PROBE.md
training/P2_SINGLE_NAV_BACKEND.md
training/README_RMAPPO.md
```

不要覆盖 `training/third_party`，否则可能再次删除已经编译的 TensorDict/TorchRL 扩展。

## 执行顺序

必须先完成 [P2_D455M_CAMERA_PROBE.md](P2_D455M_CAMERA_PROBE.md) 的相机单项验收。
相机未通过时不要运行 backend 探针。

在未加载 Isaac 的终端先跑纯 Python 合同测试：

```bash
conda activate NavRL_lxc
cd /home/lww4090/racer_ws/src/RACER-main/training

python -m pytest -q \
  training/tests/test_isaac_single_probe.py \
  training/tests/test_isaac_single_backend.py \
  training/tests/test_racer_rmappo.py
```

然后在新终端按正确顺序加载环境：

```bash
conda activate NavRL_lxc
source "$ISAACSIM_PATH/setup_conda_env.sh"
cd /home/lww4090/racer_ws/src/RACER-main/training

PYTHONUNBUFFERED=1 python -u \
  training/scripts/diagnose_isaac_navigation_backend.py \
  --steps 40 \
  --forward-action 0.25 \
  --headless \
  --output runs/isaac_single_backend/contract_report.json

echo $?
```

预期标志：

```text
P2_BACKEND_RESULT=PASS
P2_BACKEND_ENV_CLOSED
```

退出码应为 0。

## 报告判读

- `observation_shapes.depth == [1,1,3,40,64]`；
- `ego == [1,1,11]`、`target == [1,1,7]`；
- `neighbors == [1,1,5,8]`；
- `candidates == [1,1,16,9]`；
- `decision_mask == [1,1,1]` 且内容始终为 0；
- `critic_state_shape == [1,1,13]`；
- `final_command_body[0]` 应约为 `0.5 m/s`，因为 forward action 0.25 乘前进
  上限 2 m/s；
- 40 步内 `collision_steps=0`、`out_of_bounds_steps=0`、`stall_steps=0`、
  `safety_takeover_steps=0`；
- reward、所有 observation 和 critic state 均为有限值；
- `control_steps_per_second` 是单环境、单相机、CPU depth copy 的初始性能基线。

该探针不要求 40 步内到达目标，因为它测试 backend 连通性，不评估策略能力。

## 失败时的优先检查

- depth shape/数值失败：返回 D455M 单项探针，不调整奖励。
- `command_body` 不为约 0.5 m/s：检查归一化动作缩放以及正/反向非对称上限。
- reward 非有限：逐项检查 `reward_components`，重点检查深度最小距离和 target distance。
- 静止时很快 stall：检查 3 s 窗口是否按 20 Hz 得到 60 步，以及 reset 是否清空
  位置历史。
- 环境关闭崩溃：检查日志是否出现 `P2_BACKEND_ENV_CLOSED`，以及 camera annotator
  是否在 PhysX/Kit 之前释放。

## 通过后的下一步

1. 增加规则比例控制器，验证空旷目标确实能完成，并冻结一组回归轨迹。
2. 把目标移动到墙后，先用规则/人工动作验证存在可达绕行路径。
3. 增加专用 Isaac 训练入口管理 `SimulationApp` 生命周期。
4. 只训练导航头，在固定单墙场景做短连通性训练和固定场景过拟合。
5. 达到简单保留场景无碰撞成功率门槛后，再随机化树木、建筑和目标。
