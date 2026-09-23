import torch
import torch.nn.functional as F
import einops
import numpy as np
import trimesh
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
import omni.isaac.orbit.sim as sim_utils
from omni_drones.robots.drone import MultirotorBase
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni.isaac.orbit.terrains.height_field.utils import convert_height_field_to_mesh
from omni.isaac.orbit.utils import configclass
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.core.utils.viewports import set_camera_view
from utils import vec_to_new_frame, vec_to_world, construct_input
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
import time


def _range_to_tuple(value):
    return (float(value[0]), float(value[1]))


def _sample_range(value):
    low, high = _range_to_tuple(value)
    return float(np.random.uniform(low, high))


def _sequence_to_tuple(value):
    return tuple(float(item) for item in value)


def _sample_piecewise_range(range_edges, probabilities):
    range_edges = _sequence_to_tuple(range_edges)
    probabilities = np.asarray(_sequence_to_tuple(probabilities), dtype=np.float64)
    if len(range_edges) != len(probabilities) + 1:
        raise ValueError(
            "Piecewise range sampling expects len(range_edges) == len(probabilities) + 1, "
            f"got {len(range_edges)} and {len(probabilities)}."
        )
    probability_sum = probabilities.sum()
    if probability_sum <= 0.0:
        raise ValueError("Piecewise range probabilities must sum to a positive value.")
    probabilities = probabilities / probability_sum
    interval_idx = int(np.random.choice(len(probabilities), p=probabilities))
    return float(np.random.uniform(range_edges[interval_idx], range_edges[interval_idx + 1]))


def _sample_wall_height(cfg):
    if len(cfg.wall_height_range) > 2:
        return _sample_piecewise_range(cfg.wall_height_range, cfg.wall_height_probability)
    return _sample_range(cfg.wall_height_range)


def _sample_wall_length(cfg):
    if len(cfg.wall_length_range) > 2:
        return _sample_piecewise_range(cfg.wall_length_range, cfg.wall_length_probability)
    return _sample_range(cfg.wall_length_range)


def _sample_wall_center(size, max_extent, placement_margin):
    margin = max(float(placement_margin), 0.5 * float(max_extent) + 1.0)
    margin_x = min(margin, float(size[0]) * 0.45)
    margin_y = min(margin, float(size[1]) * 0.45)
    return np.array(
        [
            np.random.uniform(margin_x, float(size[0]) - margin_x),
            np.random.uniform(margin_y, float(size[1]) - margin_y),
        ],
        dtype=np.float32,
    )


def _rotation_2d(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def _wall_segment_spec(center_xy, local_offset_xy, length, thickness, height, yaw, local_yaw=0.0):
    segment_yaw = yaw + local_yaw
    segment_xy = np.asarray(center_xy) + _rotation_2d(yaw) @ np.asarray(local_offset_xy, dtype=np.float32)
    return {
        "center_xy": np.asarray(segment_xy, dtype=np.float32),
        "length": float(length),
        "thickness": float(thickness),
        "height": float(height),
        "yaw": float(segment_yaw),
    }


def _wall_segment_mesh(segment):
    center_xy = segment["center_xy"]
    length = segment["length"]
    thickness = segment["thickness"]
    height = segment["height"]
    segment_yaw = segment["yaw"]
    transform = np.eye(4)
    transform[0:3, -1] = (float(center_xy[0]), float(center_xy[1]), float(height) * 0.5)
    transform[0:3, 0:3] = np.array(
        [
            [np.cos(segment_yaw), -np.sin(segment_yaw), 0.0],
            [np.sin(segment_yaw), np.cos(segment_yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return trimesh.creation.box((float(length), float(thickness), float(height)), transform=transform)


def _make_single_wall(center, cfg):
    length = _sample_wall_length(cfg)
    thickness = _sample_range(cfg.wall_thickness_range)
    height = _sample_wall_height(cfg)
    yaw = float(np.random.uniform(0.0, 2.0 * np.pi))
    max_extent = length + thickness
    segments = [_wall_segment_spec(center, (0.0, 0.0), length, thickness, height, yaw)]
    return segments, max_extent


def _make_l_wall(center, cfg):
    arm_x = _sample_range(cfg.wall_l_length_range)
    arm_y = _sample_range(cfg.wall_l_length_range)
    thickness = _sample_range(cfg.wall_thickness_range)
    height = _sample_wall_height(cfg)
    yaw = float(np.random.uniform(0.0, 2.0 * np.pi))
    signs = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
    sign_x, sign_y = signs[int(np.random.randint(0, len(signs)))]
    max_extent = max(arm_x, arm_y) + thickness
    segments = [
        _wall_segment_spec(center, (0.0, -sign_y * arm_y * 0.5), arm_x, thickness, height, yaw),
        _wall_segment_spec(center, (-sign_x * arm_x * 0.5, 0.0), arm_y, thickness, height, yaw, np.pi * 0.5),
    ]
    return segments, max_extent


def _make_u_wall(center, cfg):
    width = _sample_range(cfg.wall_u_width_range)
    depth = _sample_range(cfg.wall_u_depth_range)
    thickness = _sample_range(cfg.wall_thickness_range)
    height = _sample_wall_height(cfg)
    opening_direction = int(np.random.randint(0, 4))
    yaw = float(np.random.uniform(0.0, 2.0 * np.pi) + opening_direction * np.pi * 0.5)
    max_extent = max(width, depth) + thickness
    segments = [
        _wall_segment_spec(center, (0.0, -depth * 0.5), width + thickness, thickness, height, yaw),
        _wall_segment_spec(center, (-width * 0.5, 0.0), depth, thickness, height, yaw, np.pi * 0.5),
        _wall_segment_spec(center, (width * 0.5, 0.0), depth, thickness, height, yaw, np.pi * 0.5),
    ]
    return segments, max_extent


def _grid_coordinates(size, horizontal_scale):
    width_pixels = int(float(size[0]) / float(horizontal_scale)) + 1
    length_pixels = int(float(size[1]) / float(horizontal_scale)) + 1
    x = np.linspace(0.0, float(size[0]), width_pixels, dtype=np.float32)
    y = np.linspace(0.0, float(size[1]), length_pixels, dtype=np.float32)
    return np.meshgrid(x, y, indexing="ij")


def _segments_mask(grid_x, grid_y, segments, clearance=0.0):
    mask = np.zeros_like(grid_x, dtype=bool)
    for segment in segments:
        center_xy = segment["center_xy"]
        rel_x = grid_x - float(center_xy[0])
        rel_y = grid_y - float(center_xy[1])
        c = float(np.cos(segment["yaw"]))
        s = float(np.sin(segment["yaw"]))
        local_x = c * rel_x + s * rel_y
        local_y = -s * rel_x + c * rel_y
        half_length = 0.5 * float(segment["length"]) + float(clearance)
        half_thickness = 0.5 * float(segment["thickness"]) + float(clearance)
        mask |= (np.abs(local_x) <= half_length) & (np.abs(local_y) <= half_thickness)
    return mask


def _generate_discrete_obstacles_height_field(cfg):
    width_pixels = int(float(cfg.size[0]) / float(cfg.horizontal_scale)) + 1
    length_pixels = int(float(cfg.size[1]) / float(cfg.horizontal_scale)) + 1
    obs_width_min = max(1, int(float(cfg.obstacle_width_range[0]) / float(cfg.horizontal_scale)))
    obs_width_max = max(obs_width_min + 1, int(float(cfg.obstacle_width_range[1]) / float(cfg.horizontal_scale)))
    platform_width = int(float(cfg.platform_width) / float(cfg.horizontal_scale))

    obs_width_range = np.arange(obs_width_min, obs_width_max + 1, 4)
    obs_length_range = np.arange(obs_width_min, obs_width_max + 1, 4)
    obs_x_range = np.arange(0, width_pixels, 4)
    obs_y_range = np.arange(0, length_pixels, 4)
    obstacles_history = []
    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.int16)
    probability_length = len(cfg.obstacle_height_probability)

    def good_distance(x, y, width, length, obstacles_hist, bad_range=(2, 10)):
        lower_bound_pixels = bad_range[0]
        upper_bound_pixels = bad_range[1]
        for (xp, yp, wp, lp) in obstacles_hist:
            dx = abs(xp - x) - width
            dy = abs(yp - y) - length
            if dx < 0 and dy < 0:
                continue
            distance = np.sqrt(dx**2 + dy**2)
            if lower_bound_pixels <= distance <= upper_bound_pixels:
                return False
        return True

    for _ in range(cfg.num_obstacles):
        if cfg.obstacle_height_mode == "choice":
            height = np.random.choice([-2, -1, 1, 2])
        elif cfg.obstacle_height_mode == "fixed":
            height = float(cfg.obstacle_height_range[0]) / float(cfg.vertical_scale)
        elif cfg.obstacle_height_mode == "range":
            random_roll = int(np.random.choice(probability_length, 1, p=cfg.obstacle_height_probability))
            low = float(cfg.obstacle_height_range[random_roll]) / float(cfg.vertical_scale)
            high = float(cfg.obstacle_height_range[random_roll + 1]) / float(cfg.vertical_scale)
            height = np.random.uniform(low, high)
        else:
            raise ValueError(
                f"Unknown obstacle height mode '{cfg.obstacle_height_mode}'. Must be 'choice', 'fixed' or 'range'."
            )

        attempts = 0
        while attempts < 100000:
            width = int(np.random.choice(obs_width_range))
            length = int(np.random.choice(obs_length_range))
            x_start = int(np.random.choice(obs_x_range))
            y_start = int(np.random.choice(obs_y_range))
            if x_start + width > width_pixels:
                x_start = width_pixels - width
            if y_start + length > length_pixels:
                y_start = length_pixels - length
            if good_distance(x_start, y_start, width, length, obstacles_history) or not obstacles_history:
                break
            attempts += 1

        obstacles_history.append((x_start, y_start, width, length))
        hf_raw[x_start : x_start + width, y_start : y_start + length] = int(np.rint(height))

    if platform_width > 0:
        x1 = (width_pixels - platform_width) // 2
        x2 = (width_pixels + platform_width) // 2
        y1 = (length_pixels - platform_width) // 2
        y2 = (length_pixels + platform_width) // 2
        hf_raw[x1:x2, y1:y2] = 0

    return hf_raw


def _sample_valid_wall_segments(wall_type, cfg, placed_centers, static_occupied, wall_occupied, grid_x, grid_y):
    for _ in range(int(cfg.wall_sampling_attempts)):
        if wall_type == "single":
            segments_builder = _make_single_wall
        elif wall_type == "l":
            segments_builder = _make_l_wall
        elif wall_type == "u":
            segments_builder = _make_u_wall
        else:
            raise ValueError(f"Unknown wall type: {wall_type}")

        provisional_center = _sample_wall_center(cfg.size, 0.0, cfg.wall_placement_margin)
        segments, _ = segments_builder(provisional_center, cfg)
        center = np.mean([segment["center_xy"] for segment in segments], axis=0)

        if any(np.linalg.norm(center - prev) < float(cfg.wall_min_separation) for prev in placed_centers):
            continue

        wall_clearance_mask = _segments_mask(grid_x, grid_y, segments, clearance=float(cfg.wall_min_separation))
        if np.any(wall_occupied & wall_clearance_mask):
            continue

        exact_mask = _segments_mask(grid_x, grid_y, segments, clearance=0.0)
        if not np.any(exact_mask):
            continue

        placed_centers.append(center)
        return segments, exact_mask

    return None, None


def _sample_valid_wall_center(cfg, placed_centers, max_extent):
    min_separation = max(float(cfg.wall_min_separation), float(max_extent))
    center = _sample_wall_center(cfg.size, max_extent, cfg.wall_placement_margin)
    for _ in range(100):
        if all(np.linalg.norm(center - prev) >= min_separation for prev in placed_centers):
            placed_centers.append(center)
            return center
        center = _sample_wall_center(cfg.size, max_extent, cfg.wall_placement_margin)
    placed_centers.append(center)
    return center


def discrete_obstacles_with_curriculum_walls_terrain(difficulty, cfg):
    heights = _generate_discrete_obstacles_height_field(cfg)
    vertices, triangles = convert_height_field_to_mesh(
        heights, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
    )
    terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    meshes = [terrain_mesh]
    origin = np.array([0.5 * float(cfg.size[0]), 0.5 * float(cfg.size[1]), 0.0], dtype=np.float32)

    static_occupied = heights > 0
    wall_occupied = np.zeros_like(static_occupied, dtype=bool)
    wall_height_map = np.zeros_like(heights, dtype=np.float32)
    grid_x, grid_y = _grid_coordinates(cfg.size, cfg.horizontal_scale)
    wall_style = int(cfg.wall_style)

    cfg._terrain_occupancy = static_occupied.copy()
    cfg._wall_occupancy = wall_occupied.copy()
    cfg._wall_height_map = wall_height_map.copy()
    cfg._terrain_size = np.array(cfg.size, dtype=np.float32)
    cfg._terrain_resolution = float(cfg.horizontal_scale)

    if wall_style == 0:
        return meshes, origin

    # Walls are appended to the terrain mesh so static LiDAR and collision use the same geometry.
    if wall_style == 1:
        wall_plan = [("single", 1)]
    elif wall_style == 2:
        wall_plan = [("single", 2)]
    elif wall_style == 3:
        wall_plan = [("single", 3)]
    elif wall_style == 4:
        wall_plan = [("single", 2), ("l", 1)]
    elif wall_style == 5:
        wall_plan = [("single", 3), ("l", 1)]
    elif wall_style == 6:
        wall_plan = [("single", 2), ("l", 1), ("u", 1)]
    elif wall_style == 7:
        wall_plan = [("single", 2), ("l", 1), ("u", 2)]
    else:
        raise ValueError(f"Unsupported env.wall_style={wall_style}. Expected an integer in [0, 7].")

    placed_centers = []
    for wall_type, wall_count in wall_plan:
        for _ in range(wall_count):
            segments, exact_mask = _sample_valid_wall_segments(
                wall_type, cfg, placed_centers, static_occupied, wall_occupied, grid_x, grid_y
            )
            if segments is None:
                continue
            meshes.extend(_wall_segment_mesh(segment) for segment in segments)
            wall_occupied |= exact_mask
            wall_height = max(float(segment["height"]) for segment in segments)
            wall_height_map[exact_mask] = np.maximum(wall_height_map[exact_mask], wall_height)

    cfg._terrain_occupancy = static_occupied | wall_occupied
    cfg._wall_occupancy = wall_occupied.copy()
    cfg._wall_height_map = wall_height_map.copy()
    return meshes, origin


@configclass
class HfDiscreteObstaclesWithWallsTerrainCfg(HfDiscreteObstaclesTerrainCfg):
    function = discrete_obstacles_with_curriculum_walls_terrain

    wall_style: int = 0
    wall_length_range: tuple[float, ...] = (5.0, 7.0, 10.0, 12.0)
    wall_length_probability: tuple[float, ...] = (0.10, 0.10, 0.80)
    wall_l_length_range: tuple[float, float] = (8.0, 10.0)
    wall_u_width_range: tuple[float, float] = (10.0, 12.0)
    wall_u_depth_range: tuple[float, float] = (3.5, 6.5)
    wall_thickness_range: tuple[float, float] = (0.35, 0.65)
    wall_height_range: tuple[float, ...] = (2.0, 4.0, 6.0)
    wall_height_probability: tuple[float, ...] = (0.0, 1.0)
    wall_placement_margin: float = 4.0
    wall_min_separation: float = 4.0
    wall_obstacle_clearance: float = 0.4
    wall_sampling_attempts: int = 500


class NavigationEnv(IsaacEnv):

    # In one step:
    # 1. _pre_sim_step (apply action) -> step isaac sim
    # 2. _post_sim_step (update lidar)
    # 3. increment progress_buf
    # 4. _compute_state_and_obs (get observation and states, update stats)
    # 5. _compute_reward_and_done (update reward and calculate returns)

    def __init__(self, cfg):
        print("[Navigation Environment]: Initializing Env...")
        # LiDAR params:
        self.lidar_range = cfg.sensor.lidar_range
        self.lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))
        self.lidar_vbeams = cfg.sensor.lidar_vbeams
        self.lidar_hres = cfg.sensor.lidar_hres
        self.lidar_hbeams = int(360/self.lidar_hres)
        stuck_cfg = cfg.get("stuck", {})
        self.stuck_window = max(1, int(stuck_cfg.get("window", 40)))
        self.stuck_progress_eps = float(stuck_cfg.get("progress_eps", 0.005))
        self.stuck_front_distance = min(self.lidar_range, float(stuck_cfg.get("front_obstacle_distance", 1.5)))
        self.stuck_front_tan = float(np.tan(np.deg2rad(stuck_cfg.get("front_angle_deg", 35.0))))
        self.stuck_front_height = float(stuck_cfg.get("front_height", 0.75))
        reward_cfg = cfg.get("reward", {})
        goal_progress_cfg = reward_cfg.get("goal_progress", {})
        self.goal_radius = float(goal_progress_cfg.get("goal_radius", 0.5))
        self.goal_arrival_reward = float(goal_progress_cfg.get("arrival_reward", 5.0))
        self.goal_timeout_penalty = float(goal_progress_cfg.get("timeout_penalty", self.goal_arrival_reward))
        self.goal_progress_scale = float(goal_progress_cfg.get("progress_scale", 10.0))
        stall_reward_cfg = reward_cfg.get("stall", {})
        self.stall_reward_weight = float(stall_reward_cfg.get("weight", 0.5))
        self.stall_reward_window = max(1, int(stall_reward_cfg.get("window", self.stuck_window)))
        self.stall_reward_ramp_window = max(1, int(stall_reward_cfg.get("ramp_window", self.stall_reward_window)))
        self.stall_distance_threshold = float(stall_reward_cfg.get("distance_threshold", 0.25))
        self.stall_speed_threshold = float(stall_reward_cfg.get("speed_threshold", 0.15))
        self.stall_progress_threshold = float(stall_reward_cfg.get("progress_eps", self.stuck_progress_eps))
        escape_reward_cfg = reward_cfg.get("escape", {})
        self.escape_reward_weight = float(escape_reward_cfg.get("weight", 1.0))
        self.escape_recovery_window = max(1, int(escape_reward_cfg.get("recovery_window", self.stuck_window * 3)))
        self.escape_cooldown_steps = max(0, int(escape_reward_cfg.get("cooldown_steps", self.stuck_window * 2)))
        self.escape_min_displacement = max(float(escape_reward_cfg.get("min_displacement", 0.3)), 0.0)
        self.escape_min_goal_progress = max(float(escape_reward_cfg.get("min_goal_progress", 0.2)), 0.0)
        self.escape_min_clearance_gain = max(float(escape_reward_cfg.get("min_clearance_gain", 0.2)), 0.0)
        vo_cfg = reward_cfg.get("vo", {})
        self.vo_weight = float(vo_cfg.get("weight", 1.0))
        self.vo_tau = max(float(vo_cfg.get("tau", 0.75)), 1e-6)
        self.vo_horizon = max(float(vo_cfg.get("horizon", 2.5)), 1e-6)
        self.vo_xy_margin = float(vo_cfg.get("xy_margin", 0.3))
        self.vo_z_margin = float(vo_cfg.get("z_margin", 0.3))
        self.vo_action_weight = float(vo_cfg.get("action_weight", 0.0))
        self.vo_static_weight = float(vo_cfg.get("static_weight", 0.0))
        self.vo_safe_action_weight = float(vo_cfg.get("safe_action_weight", 0.0))
        self.vo_static_radius = max(float(vo_cfg.get("static_radius", 0.0)), 0.0)
        self.vo_static_vertical_min = float(vo_cfg.get("static_vertical_min", -0.35))
        self.vo_reward_topk = max(1, int(vo_cfg.get("reward_topk", 10)))
        self.vo_sphere_stack = bool(vo_cfg.get("sphere_stack", False))
        self.vo_sphere_stack_max_spheres = int(vo_cfg.get("sphere_stack_max_spheres", 0))
        self.vo_sphere_stack_z_margin = max(float(vo_cfg.get("sphere_stack_z_margin", 0.6)), 0.0)
        self.vo_reward_range = max(self.lidar_range, float(max(cfg.env_dyn.local_range)))
        self.vo_warmup_steps = max(0, int(vo_cfg.get("warmup_steps", 0)))
        self.vo_step_count = 0
        noise_cfg = cfg.get("observation_noise", {})
        self.obs_noise_enabled = bool(noise_cfg.get("enabled", False))
        self.obs_noise_train_only = bool(noise_cfg.get("train_only", True))
        lidar_noise_cfg = noise_cfg.get("lidar", {})
        self.lidar_noise_std = max(float(lidar_noise_cfg.get("std", 0.0)), 0.0)
        self.lidar_dropout_prob = min(max(float(lidar_noise_cfg.get("dropout_prob", 0.0)), 0.0), 1.0)
        self.lidar_scale_std = max(float(lidar_noise_cfg.get("scale_std", 0.0)), 0.0)
        self.lidar_quantization = max(float(lidar_noise_cfg.get("quantization", 0.0)), 0.0)
        dyn_obs_noise_cfg = noise_cfg.get("dynamic_obstacle", {})
        self.dyn_obs_pos_noise_std = max(float(dyn_obs_noise_cfg.get("position_std", 0.0)), 0.0)
        self.dyn_obs_vel_noise_std = max(float(dyn_obs_noise_cfg.get("velocity_std", 0.0)), 0.0)
        self.dyn_obs_size_noise_std = max(float(dyn_obs_noise_cfg.get("size_std", 0.0)), 0.0)
        self.dyn_obs_dropout_prob = min(max(float(dyn_obs_noise_cfg.get("dropout_prob", 0.0)), 0.0), 1.0)
        self.dyn_obs_frame_dropout_prob = min(max(float(dyn_obs_noise_cfg.get("frame_dropout_prob", 0.0)), 0.0), 1.0)
        obs_delay_cfg = noise_cfg.get("observation_delay", {})
        action_delay_cfg = noise_cfg.get("action_delay", {})
        action_execution_cfg = noise_cfg.get("action_execution", {})
        self.observation_delay_steps = max(int(obs_delay_cfg.get("steps", 0)), 0)
        self.action_delay_steps = max(int(action_delay_cfg.get("steps", 0)), 0)
        self.action_execution_noise_std = max(float(action_execution_cfg.get("std", 0.0)), 0.0)
        self.action_execution_scale_std = max(float(action_execution_cfg.get("scale_std", 0.0)), 0.0)
        self.observation_delay_buffers = None
        self.action_delay_buffer = None
        self.prev_dyn_obs_states = None

        task_sampling_cfg = cfg.env.get("task_sampling", {})
        self.task_boundary_coordinate = float(task_sampling_cfg.get("boundary_coordinate", 21.0))
        self.task_tangent_limit = float(task_sampling_cfg.get("tangent_limit", 18.0))
        self.task_target_jitter = max(float(task_sampling_cfg.get("target_jitter", 4.0)), 0.0)
        self.horizontal_limit = float(task_sampling_cfg.get("horizontal_limit", 22.0))
        self.task_dynamic_clearance = max(
            float(task_sampling_cfg.get("dynamic_clearance_radius", 1.5)), 0.0
        )
        self.wall_influence_radius = max(float(task_sampling_cfg.get("wall_influence_radius", 2.0)), 0.0)
        self.wall_line_sample_spacing = max(
            float(task_sampling_cfg.get("wall_line_sample_spacing", 0.25)), 0.05
        )
        if self.task_boundary_coordinate <= 0.0:
            raise ValueError("env.task_sampling.boundary_coordinate must be positive.")
        if not 0.0 < self.task_tangent_limit <= self.task_boundary_coordinate:
            raise ValueError(
                "env.task_sampling.tangent_limit must be positive and no larger than boundary_coordinate."
            )
        if self.horizontal_limit <= self.task_boundary_coordinate:
            raise ValueError(
                "env.task_sampling.horizontal_limit must be larger than boundary_coordinate."
            )

        super().__init__(cfg, cfg.headless)
        
        # Drone Initialization
        self.drone.initialize()
        self.init_vels = torch.zeros_like(self.drone.get_velocities())


        # LiDAR Intialization
        ray_caster_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            # attach_yaw_only=False,
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=self.lidar_hres, # horizontal default is set to 10
                vertical_ray_angles=torch.linspace(*self.lidar_vfov, self.lidar_vbeams) 
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
            # mesh_prim_paths=["/World"],
        )
        self.lidar = RayCaster(ray_caster_cfg)
        self.lidar._initialize_impl()
        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams) 
        
        # start and target 
        with torch.device(self.device):
            # self.start_pos = torch.zeros(self.num_envs, 1, 3)
            self.target_pos = torch.zeros(self.num_envs, 1, 3)
            
            # Coordinate change: add target direction variable
            self.target_dir = torch.zeros(self.num_envs, 1, 3)
            self.height_range = torch.zeros(self.num_envs, 1, 2)
            self.prev_drone_vel_w = torch.zeros(self.num_envs, 1 , 3)
            self.prev_goal_distance = torch.zeros(self.num_envs, 1)
            self.prev_front_clearance = torch.full((self.num_envs, 1), self.lidar_range)
            self.stuck_counter = torch.zeros(self.num_envs, 1, dtype=torch.long)
            self.escape_armed = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.escape_age = torch.zeros(self.num_envs, 1, dtype=torch.long)
            self.escape_cooldown = torch.zeros(self.num_envs, 1, dtype=torch.long)
            self.escape_anchor_pos = torch.zeros(self.num_envs, 1, 3)
            self.escape_anchor_goal_distance = torch.zeros(self.num_envs, 1)
            self.escape_anchor_clearance = torch.full((self.num_envs, 1), self.lidar_range)
            self.stall_counter = torch.zeros(self.num_envs, 1, dtype=torch.long)
            self.stall_anchor_pos = torch.zeros(self.num_envs, 1, 3)
            self.observation_delay_reset_mask = torch.zeros(self.num_envs, dtype=torch.bool)
            self.action_delay_reset_mask = torch.zeros(self.num_envs, dtype=torch.bool)
            self.dyn_obs_frame_drop_reset_mask = torch.zeros(self.num_envs, dtype=torch.bool)
            self.train_task_mode = self._normalize_task_mode(
                str(self.cfg.get("train_style", "opposite_crossing"))
            )
            self.eval_task_mode = "opposite_crossing"
            # self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            # self.target_pos[:, 0, 1] = 24.
            # self.target_pos[:, 0, 2] = 2.     


    def _design_scene(self):
        # Initialize a drone in prim /World/envs/envs_0
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name] # drone model class
        cfg = drone_model.cfg_cls(force_sensor=False)
        self.drone = drone_model(cfg=cfg)
        # drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 1.0)])[0]
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]

        # lighting
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)
        
        # Ground Plane
        cfg_ground = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300., 300.))
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

        self.map_range = [20.0, 20.0, 4.5]
        wall_cfg = self.cfg.env.get("wall", {})
        self.terrain_occupancy = None
        self.wall_occupancy = None
        self.wall_height_map = None
        self.terrain_occupancy_resolution = 0.1
        self.terrain_occupancy_size = torch.tensor(
            [self.map_range[0] * 2.0, self.map_range[1] * 2.0], dtype=torch.float32, device=self.device
        )
        self.terrain_clearance_cache = {}
        self.wall_collision_cache = {}
        self.wall_influence_cache = {}
        self.spawn_clearance_radius = float(wall_cfg.get("spawn_clearance_radius", 1.5))
        self.target_clearance_radius = float(wall_cfg.get("target_clearance_radius", self.spawn_clearance_radius))
        self.position_sample_attempts = max(1, int(wall_cfg.get("position_sample_attempts", 128)))

        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(self.map_range[0]*2, self.map_range[1]*2), 
                border_width=5.0,
                num_rows=1, 
                num_cols=1, 
                horizontal_scale=0.1,
                vertical_scale=0.1,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height",
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesWithWallsTerrainCfg(
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=self.cfg.env.num_obstacles,
                        obstacle_height_mode="range",
                        obstacle_width_range=(0.4, 1.1),
                        obstacle_height_range=[1.0, 1.5, 2.0, 4.0, 6.0],
                        obstacle_height_probability=[0.1, 0.15, 0.20, 0.55],
                        platform_width=0.0,
                        wall_style=int(self.cfg.env.get("wall_style", 0)),
                        wall_length_range=_sequence_to_tuple(wall_cfg.get("length_range", [5.0, 7.0, 10.0, 12.0])),
                        wall_length_probability=_sequence_to_tuple(wall_cfg.get("length_probability", [0.10, 0.10, 0.80])),
                        wall_l_length_range=_range_to_tuple(wall_cfg.get("l_length_range", [8.0, 10.0])),
                        wall_u_width_range=_range_to_tuple(wall_cfg.get("u_width_range", [10.0, 12.0])),
                        wall_u_depth_range=_range_to_tuple(wall_cfg.get("u_depth_range", [3.5, 6.5])),
                        wall_thickness_range=_range_to_tuple(wall_cfg.get("thickness_range", [0.35, 0.65])),
                        wall_height_range=_sequence_to_tuple(wall_cfg.get("height_range", [2.0, 4.0, 6.0])),
                        wall_height_probability=_sequence_to_tuple(wall_cfg.get("height_probability", [0.0, 1.0])),
                        wall_placement_margin=float(wall_cfg.get("placement_margin", 4.0)),
                        wall_min_separation=float(wall_cfg.get("min_separation", 4.0)),
                        wall_obstacle_clearance=float(wall_cfg.get("obstacle_clearance", 0.4)),
                        wall_sampling_attempts=int(wall_cfg.get("sampling_attempts", 500)),
                    ),
                },
            ),
            visual_material = None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=True,
        )
        terrain_importer = TerrainImporter(terrain_cfg)
        obstacle_terrain_cfg = terrain_cfg.terrain_generator.sub_terrains["obstacles"]
        terrain_occupancy = getattr(obstacle_terrain_cfg, "_terrain_occupancy", None)
        wall_occupancy = getattr(obstacle_terrain_cfg, "_wall_occupancy", None)
        wall_height_map = getattr(obstacle_terrain_cfg, "_wall_height_map", None)
        terrain_size = getattr(obstacle_terrain_cfg, "_terrain_size", None)
        terrain_resolution = getattr(obstacle_terrain_cfg, "_terrain_resolution", None)
        if terrain_occupancy is not None:
            self.terrain_occupancy = torch.as_tensor(
                terrain_occupancy.astype(np.float32), device=self.device, dtype=torch.float32
            )
        if wall_occupancy is not None:
            self.wall_occupancy = torch.as_tensor(
                wall_occupancy.astype(np.float32), device=self.device, dtype=torch.float32
            )
        if wall_height_map is not None:
            self.wall_height_map = torch.as_tensor(
                wall_height_map.astype(np.float32), device=self.device, dtype=torch.float32
            )
        if terrain_size is not None:
            self.terrain_occupancy_size = torch.as_tensor(terrain_size[:2], device=self.device, dtype=torch.float32)
        if terrain_resolution is not None:
            self.terrain_occupancy_resolution = float(terrain_resolution)

        if (self.cfg.env_dyn.num_obstacles == 0):
            return
        # Dynamic Obstacles
        # NOTE: we use cuboid to represent 3D dynamic obstacles which can float in the air 
        # and the long cylinder to represent 2D dynamic obstacles for which the drone can only pass in 2D 
        # The width of the dynamic obstacles is divided into N_w=4 bins
        # [[0, 0.25], [0.25, 0.50], [0.50, 0.75], [0.75, 1.0]]
        # The height of the dynamic obstacles is divided into N_h=2 bins
        # [[0, 0.5], [0.5, inf]] we want to distinguish 3D obstacles and 2d obstacles
        N_w = 4 # number of width intervals between [0, 1]
        N_h = 2 # number of height: current only support binary
        max_obs_width = 1.0
        self.max_obs_3d_height = 1.0
        self.max_obs_2d_height = 5.0
        self.dyn_obs_width_res = max_obs_width/float(N_w)
        dyn_obs_category_num = N_w * N_h
        self.dyn_obs_num_of_each_category = int(self.cfg.env_dyn.num_obstacles / dyn_obs_category_num)
        self.cfg.env_dyn.num_obstacles = self.dyn_obs_num_of_each_category * dyn_obs_category_num # in case of the roundup error


        # Dynamic obstacle info
        self.dyn_obs_list = []
        self.dyn_obs_state = torch.zeros((self.cfg.env_dyn.num_obstacles, 13), dtype=torch.float, device=self.cfg.device) # 13 is based on the states from sim, we only care the first three which is position
        self.dyn_obs_state[:, 3] = 1. # Quaternion
        self.dyn_obs_goal = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_origin = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_vel = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_step_count = 0 # dynamic obstacle motion step count
        self.dyn_obs_size = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.device) # size of dynamic obstacles


        # helper function to check pos validity for even distribution condition
        def check_pos_validity(prev_pos_list, curr_pos, adjusted_obs_dist):
            for prev_pos in prev_pos_list:
                if (np.linalg.norm(curr_pos - prev_pos) <= adjusted_obs_dist):
                    return False
            return True            
        
        obs_dist = 2 * np.sqrt(self.map_range[0] * self.map_range[1] / self.cfg.env_dyn.num_obstacles) # prefered distance between each dynamic obstacle
        curr_obs_dist = obs_dist
        prev_pos_list = [] # for distance check
        cuboid_category_num = cylinder_category_num = int(dyn_obs_category_num/N_h)
        for category_idx in range(cuboid_category_num + cylinder_category_num):
            # create all origins for 3D dynamic obstacles of this category (size)
            for origin_idx in range(self.dyn_obs_num_of_each_category):
                # random sample an origin until satisfy the evenly distributed condition
                start_time = time.time()
                while (True):
                    ox = np.random.uniform(low=-self.map_range[0], high=self.map_range[0])
                    oy = np.random.uniform(low=-self.map_range[1], high=self.map_range[1])
                    if (category_idx < cuboid_category_num):
                        oz = np.random.uniform(low=0.0, high=self.map_range[2]) 
                    else:
                        oz = self.max_obs_2d_height/2. # half of the height
                    curr_pos = np.array([ox, oy])
                    valid = check_pos_validity(prev_pos_list, curr_pos, curr_obs_dist)
                    curr_time = time.time()
                    if (curr_time - start_time > 0.1):
                        curr_obs_dist *= 0.8
                        start_time = time.time()
                    if (valid):
                        prev_pos_list.append(curr_pos)
                        break
                curr_obs_dist = obs_dist
                origin = [ox, oy, oz]
                self.dyn_obs_origin[origin_idx+category_idx*self.dyn_obs_num_of_each_category] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)     
                self.dyn_obs_state[origin_idx+category_idx*self.dyn_obs_num_of_each_category, :3] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)                        
                prim_utils.create_prim(f"/World/Origin{origin_idx+category_idx*self.dyn_obs_num_of_each_category}", "Xform", translation=origin)

            # Spawn various sizes of dynamic obstacles 
            if (category_idx < cuboid_category_num):
                # spawn for 3D dynamic obstacles
                obs_width = width = float(category_idx+1) * max_obs_width/float(N_w)
                obs_height = self.max_obs_3d_height
                cuboid_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cuboid",
                    spawn=sim_utils.CuboidCfg(
                        size=[width, width, self.max_obs_3d_height],
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cuboid_cfg)
            else:
                radius = float(category_idx-cuboid_category_num+1) * max_obs_width/float(N_w) / 2.
                obs_width = radius * 2
                obs_height = self.max_obs_2d_height
                # spawn for 2D dynamic obstacles
                cylinder_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cylinder",
                    spawn=sim_utils.CylinderCfg(
                        radius = radius,
                        height = self.max_obs_2d_height, 
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cylinder_cfg)
            self.dyn_obs_list.append(dynamic_obstacle)
            self.dyn_obs_size[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category] \
                = torch.tensor([obs_width, obs_width, obs_height], dtype=torch.float, device=self.cfg.device)



    def move_dynamic_obstacle(self):
        # Step 1: Random sample new goals for required update dynamic obstacles
        # Check whether the current dynamic obstacles need new goals
        dyn_obs_goal_dist = torch.sqrt(torch.sum((self.dyn_obs_state[:, :3] - self.dyn_obs_goal)**2, dim=1)) if self.dyn_obs_step_count !=0 \
            else torch.zeros(self.dyn_obs_state.size(0), device=self.cfg.device)
        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5 # change to a new goal if less than the threshold
        
        # sample new goals in local range
        num_new_goal = torch.sum(dyn_obs_new_goal_mask)
        sample_x_local = -self.cfg.env_dyn.local_range[0] + 2. * self.cfg.env_dyn.local_range[0] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_y_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[1] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_z_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[2] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_goal_local = torch.cat([sample_x_local, sample_y_local, sample_z_local], dim=1)
    
        # apply local goal to the global range
        self.dyn_obs_goal[dyn_obs_new_goal_mask] = self.dyn_obs_origin[dyn_obs_new_goal_mask] + sample_goal_local
        # clamp the range if out of the static env range
        self.dyn_obs_goal[:, 0] = torch.clamp(self.dyn_obs_goal[:, 0], min=-self.map_range[0], max=self.map_range[0])
        self.dyn_obs_goal[:, 1] = torch.clamp(self.dyn_obs_goal[:, 1], min=-self.map_range[1], max=self.map_range[1])
        self.dyn_obs_goal[:, 2] = torch.clamp(self.dyn_obs_goal[:, 2], min=0., max=self.map_range[2])
        self.dyn_obs_goal[int(self.dyn_obs_goal.size(0)/2):, 2] = self.max_obs_2d_height/2. # for 2d obstacles


        # Step 2: Random sample velocity for roughly every 2 seconds
        if (self.dyn_obs_step_count % int(2.0/self.cfg.sim.dt) == 0):
            self.dyn_obs_vel_norm = self.cfg.env_dyn.vel_range[0] + (self.cfg.env_dyn.vel_range[1] \
              - self.cfg.env_dyn.vel_range[0]) * torch.rand(self.dyn_obs_vel.size(0), 1, dtype=torch.float, device=self.cfg.device)
            self.dyn_obs_vel = self.dyn_obs_vel_norm * \
                (self.dyn_obs_goal - self.dyn_obs_state[:, :3])/torch.norm((self.dyn_obs_goal - self.dyn_obs_state[:, :3]), dim=1, keepdim=True)

        # Step 3: Calculate new position update for current timestep
        self.dyn_obs_state[:, :3] += self.dyn_obs_vel * self.cfg.sim.dt


        # Step 4: Update Visualized Location in Simulation
        for category_idx, dynamic_obstacle in enumerate(self.dyn_obs_list):
            dynamic_obstacle.write_root_state_to_sim(self.dyn_obs_state[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category]) 
            dynamic_obstacle.write_data_to_sim()
            dynamic_obstacle.update(self.cfg.sim.dt)

        self.dyn_obs_step_count += 1

    def _in_stuck_front_region(self, rel_pos, lateral_inflation=None, vertical_inflation=None):
        forward = rel_pos[..., 0]
        lateral = rel_pos[..., 1].abs()
        vertical = rel_pos[..., 2].abs()
        horizontal_distance = rel_pos[..., :2].norm(dim=-1)

        if lateral_inflation is None:
            lateral_inflation = 0.0
        if vertical_inflation is None:
            vertical_inflation = 0.0

        return (
            (forward > 0.0)
            & (horizontal_distance <= self.stuck_front_distance + lateral_inflation)
            & (lateral <= self.stuck_front_tan * forward.clamp_min(1e-6) + lateral_inflation)
            & (vertical <= self.stuck_front_height + vertical_inflation)
        )

    def _compute_goal_progress_reward(self, goal_progress, reach_goal, time_limit):
        reward_goal_progress = self.goal_progress_scale * goal_progress
        reward_goal_progress = torch.where(
            time_limit,
            torch.full_like(reward_goal_progress, -self.goal_timeout_penalty),
            reward_goal_progress,
        )
        reward_goal_progress = torch.where(
            reach_goal,
            torch.full_like(reward_goal_progress, self.goal_arrival_reward),
            reward_goal_progress,
        )
        return reward_goal_progress

    def _compute_escape_reward(
        self,
        previous_stuck_counter,
        current_pos,
        goal_distance,
        front_clearance,
        blocked_low_progress,
        reach_goal,
        time_limit,
    ):
        self.escape_cooldown = (self.escape_cooldown - 1).clamp_min(0)

        crossed_stuck_threshold = (
            (previous_stuck_counter < self.stuck_window)
            & (self.stuck_counter >= self.stuck_window)
        )
        arm_now = crossed_stuck_threshold & (~self.escape_armed) & (self.escape_cooldown == 0)
        self.escape_anchor_pos = torch.where(
            arm_now.unsqueeze(-1),
            current_pos,
            self.escape_anchor_pos,
        )
        self.escape_anchor_goal_distance = torch.where(
            arm_now,
            goal_distance,
            self.escape_anchor_goal_distance,
        )
        self.escape_anchor_clearance = torch.where(
            arm_now,
            front_clearance,
            self.escape_anchor_clearance,
        )
        self.escape_armed = self.escape_armed | arm_now
        self.escape_age = torch.where(
            arm_now,
            torch.zeros_like(self.escape_age),
            torch.where(self.escape_armed, self.escape_age + 1, torch.zeros_like(self.escape_age)),
        )

        displacement = (current_pos[..., :2] - self.escape_anchor_pos[..., :2]).norm(dim=-1)
        goal_progress_since_block = self.escape_anchor_goal_distance - goal_distance
        clearance_gain = front_clearance - self.escape_anchor_clearance
        recovery_evidence = (
            (displacement >= self.escape_min_displacement)
            & (
                (goal_progress_since_block >= self.escape_min_goal_progress)
                | (clearance_gain >= self.escape_min_clearance_gain)
            )
        )
        recovered = (
            self.escape_armed
            & (~blocked_low_progress)
            & recovery_evidence
            & (~reach_goal)
            & (~time_limit)
        )
        reward_escape = self.escape_reward_weight * recovered.float()

        expired = self.escape_armed & (self.escape_age >= self.escape_recovery_window)
        disarm = recovered | expired | reach_goal | time_limit
        self.escape_armed = self.escape_armed & (~disarm)
        self.escape_age = torch.where(disarm, torch.zeros_like(self.escape_age), self.escape_age)
        self.escape_cooldown = torch.where(
            recovered,
            torch.full_like(self.escape_cooldown, self.escape_cooldown_steps),
            self.escape_cooldown,
        )
        return reward_escape

    def _compute_stall_reward(self, current_pos, current_vel, goal_progress, reach_goal):
        stall_displacement = (current_pos[..., :2] - self.stall_anchor_pos[..., :2]).norm(dim=-1)
        stall_speed = current_vel.norm(dim=-1)
        low_progress = goal_progress.abs() <= self.stall_progress_threshold
        low_motion = (stall_displacement <= self.stall_distance_threshold) & (stall_speed <= self.stall_speed_threshold)
        stall_candidate = low_motion & low_progress & (~reach_goal)

        self.stall_counter = torch.where(
            stall_candidate,
            self.stall_counter + 1,
            torch.zeros_like(self.stall_counter),
        )
        self.stall_anchor_pos = torch.where(
            stall_candidate.unsqueeze(-1),
            self.stall_anchor_pos,
            current_pos.clone(),
        )

        stall_active = self.stall_counter >= self.stall_reward_window
        stall_excess = (self.stall_counter - self.stall_reward_window).clamp(min=0).float()
        stall_scale = (stall_excess / float(self.stall_reward_ramp_window)).clamp(max=1.0)
        reward_stall = -self.stall_reward_weight * stall_scale
        return reward_stall, stall_active

    def _compute_vo_scale(self, dyn_obs_size):
        effective_radius_xy = dyn_obs_size[..., 0].unsqueeze(-1) / 2.0 + self.vo_xy_margin
        effective_radius_z = dyn_obs_size[..., 2].unsqueeze(-1) / 2.0 + self.vo_z_margin
        scale = torch.cat(
            [effective_radius_xy, effective_radius_xy, effective_radius_z],
            dim=-1,
        ).clamp_min(1e-6)
        return scale

    def _expand_dynamic_obstacles_for_vo(self, obs_rpos, obs_vel, obs_size, obs_range_mask, agent_vel=None):
        if (
            not self.vo_sphere_stack
            or obs_rpos.size(1) == 0
        ):
            return obs_rpos, obs_vel, obs_size, obs_range_mask

        batch_size, obs_count, _ = obs_rpos.shape
        dtype = obs_rpos.dtype
        device = obs_rpos.device

        xy_extent = torch.maximum(obs_size[..., 0], obs_size[..., 1]).clamp_min(1e-6)
        height = obs_size[..., 2].clamp_min(1e-6)
        base_radius = obs_size[..., :2].norm(dim=-1).mul(0.5).clamp_min(1e-6)
        num_spheres = torch.ceil(height / xy_extent).clamp_min(1).long()
        if self.vo_sphere_stack_max_spheres > 0:
            num_spheres = num_spheres.clamp(max=self.vo_sphere_stack_max_spheres)
        sphere_count = int(num_spheres.max().item())

        sphere_ids = torch.arange(sphere_count, device=device).view(1, 1, sphere_count)
        valid_sphere = sphere_ids < num_spheres.unsqueeze(-1)
        num_spheres_f = num_spheres.to(dtype=dtype).clamp_min(1.0)
        step_z = xy_extent
        z_remain = height - (num_spheres_f - 1.0) * step_z

        z_offsets = (
            -0.5 * height.unsqueeze(-1)
            + 0.5 * z_remain.unsqueeze(-1)
            + sphere_ids.to(dtype=dtype) * step_z.unsqueeze(-1)
        )
        z_offsets = torch.where(valid_sphere, z_offsets, torch.zeros_like(z_offsets))

        expanded_rpos = obs_rpos.unsqueeze(2).expand(batch_size, obs_count, sphere_count, 3).clone()
        expanded_rpos[..., 2] = expanded_rpos[..., 2] + z_offsets
        expanded_vel = obs_vel.unsqueeze(2).expand(batch_size, obs_count, sphere_count, 3).clone()
        expanded_size = (2.0 * base_radius).unsqueeze(-1).unsqueeze(-1).expand(
            batch_size,
            obs_count,
            sphere_count,
            3,
        ).clone()
        expanded_mask = obs_range_mask.unsqueeze(-1) | (~valid_sphere)

        if self.vo_sphere_stack_z_margin > 0.0 and agent_vel is not None:
            agent_vel = self._format_vo_velocity(agent_vel)
            agent_vel_z = agent_vel[..., 2].reshape(batch_size, 1, 1)
            rel_z_now = expanded_rpos[..., 2]
            rel_z_future = rel_z_now + (expanded_vel[..., 2] - agent_vel_z) * self.vo_horizon
            rel_z_min = torch.minimum(rel_z_now, rel_z_future)
            rel_z_max = torch.maximum(rel_z_now, rel_z_future)
            effective_z_margin = self.vo_sphere_stack_z_margin + 0.5 * expanded_size[..., 2]
            height_relevant = (rel_z_min <= effective_z_margin) & (rel_z_max >= -effective_z_margin)
            expanded_mask = expanded_mask | (~height_relevant)

        return (
            expanded_rpos.reshape(batch_size, obs_count * sphere_count, 3),
            expanded_vel.reshape(batch_size, obs_count * sphere_count, 3),
            expanded_size.reshape(batch_size, obs_count * sphere_count, 3),
            expanded_mask.reshape(batch_size, obs_count * sphere_count),
        )

    def _vo_warmup_like(self, reference: torch.Tensor):
        if self.vo_warmup_steps > 0:
            warmup = min(float(self.vo_step_count) / float(self.vo_warmup_steps), 1.0)
        else:
            warmup = 1.0
        return torch.full_like(reference, warmup)

    def _format_vo_velocity(self, velocity: torch.Tensor):
        if velocity.dim() == 2:
            velocity = velocity.unsqueeze(1)
        return velocity[..., :3]

    def _compute_vo_risk(self, obs_rpos, obs_vel, obs_size, obs_range_mask, agent_vel):
        if obs_rpos.size(1) == 0:
            vo_risk = torch.zeros(obs_rpos.size(0), 1, dtype=obs_rpos.dtype, device=obs_rpos.device)
            risk_each = torch.zeros(obs_rpos.size(0), 0, 1, dtype=obs_rpos.dtype, device=obs_rpos.device)
            return vo_risk, risk_each

        agent_vel = self._format_vo_velocity(agent_vel)
        agent_vel_expanded = agent_vel.expand(-1, obs_rpos.size(1), -1)
        # obs_rpos is obstacle position minus drone position, so its time derivative is v_obs - v_drone.
        rel_vel = obs_vel - agent_vel_expanded

        # Scale the 3D relative motion by an anisotropic safety ellipsoid.
        scale = self._compute_vo_scale(obs_size)

        scaled_rel_pos = obs_rpos / scale
        scaled_rel_vel = rel_vel / scale

        quad_a = (scaled_rel_vel * scaled_rel_vel).sum(dim=-1, keepdim=True)
        quad_b = 2.0 * (scaled_rel_pos * scaled_rel_vel).sum(dim=-1, keepdim=True)
        quad_c = (scaled_rel_pos * scaled_rel_pos).sum(dim=-1, keepdim=True) - 1.0
        discriminant = quad_b.square() - 4.0 * quad_a * quad_c

        valid_mask = (~obs_range_mask).unsqueeze(-1)
        approaching = quad_b < 0.0
        overlapping = quad_c <= 0.0
        discriminant_positive = discriminant >= 0.0
        moving = quad_a > 1e-6
        ttc = (-quad_b - torch.sqrt(discriminant.clamp_min(0.0))) / (2.0 * quad_a.clamp_min(1e-6))
        valid_ttc = valid_mask & approaching & discriminant_positive & moving & (~overlapping) & (ttc > 0.0) & (ttc <= self.vo_horizon)

        risk_each = torch.zeros_like(ttc)
        risk_each = torch.where(valid_ttc, torch.exp(-ttc / self.vo_tau), risk_each)
        risk_each = torch.where(valid_mask & overlapping, torch.ones_like(risk_each), risk_each)
        vo_risk = risk_each.max(dim=1, keepdim=True).values.squeeze(-1)
        return vo_risk, risk_each

    def _compute_vo_reward(self, obs_rpos, obs_vel, obs_size, obs_range_mask, agent_vel, weight=None):
        vo_risk, risk_each = self._compute_vo_risk(obs_rpos, obs_vel, obs_size, obs_range_mask, agent_vel)
        vo_warmup = self._vo_warmup_like(vo_risk)
        reward_weight = self.vo_weight if weight is None else weight
        reward_vo = -reward_weight * vo_warmup * vo_risk
        return reward_vo, vo_risk, vo_warmup, risk_each

    def _compute_safe_action_penalty(self, action_vel, obs_rpos, obs_vel, obs_size, obs_range_mask):
        action_vel = self._format_vo_velocity(action_vel)
        vo_risk, risk_each = self._compute_vo_risk(obs_rpos, obs_vel, obs_size, obs_range_mask, action_vel)
        if obs_rpos.size(1) == 0:
            return torch.zeros_like(vo_risk), vo_risk

        valid_mask = (~obs_range_mask).unsqueeze(-1)
        relative_speed = action_vel.expand(-1, obs_rpos.size(1), -1) - obs_vel
        obs_distance = obs_rpos.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        obs_dir = obs_rpos / obs_distance
        closing_speed = (relative_speed * obs_dir).sum(dim=-1, keepdim=True).clamp_min(0.0)
        correction_each = torch.where(valid_mask, risk_each * closing_speed * obs_dir, torch.zeros_like(obs_rpos))

        worst_idx = risk_each.squeeze(-1).argmax(dim=1)
        correction = torch.gather(correction_each, 1, worst_idx.view(-1, 1, 1).expand(-1, 1, 3))
        action_limit = max(float(getattr(self.cfg.algo.actor, "action_limit", 1.0)), 1e-6)
        penalty = (correction.norm(dim=-1, keepdim=True) / action_limit).square().clamp(max=1.0).squeeze(1)
        penalty = penalty * (vo_risk > 0.0).float()
        return penalty, vo_risk


    def _set_specs(self):
        observation_dim = 8
        num_dim_each_dyn_obs_state = 10

        # Observation Spec
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device), 
                    "lidar": UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.lidar_vbeams), device=self.device),
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state), device=self.device),
                }),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        
        # Action Spec
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec, # number of motor
            })
        }).expand(self.num_envs).to(self.device)
        
        # Reward Spec
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)

        # Done Spec
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device) 


        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "goal_distance": UnboundedContinuousTensorSpec(1),
            "goal_progress": UnboundedContinuousTensorSpec(1),
            "front_obstacle": UnboundedContinuousTensorSpec(1),
            "front_clearance": UnboundedContinuousTensorSpec(1),
            "front_clearance_gain": UnboundedContinuousTensorSpec(1),
            "reward_goal_progress": UnboundedContinuousTensorSpec(1),
            "penalty_safety_static": UnboundedContinuousTensorSpec(1),
            "penalty_safety_dynamic": UnboundedContinuousTensorSpec(1),
            "reward_vel": UnboundedContinuousTensorSpec(1),
            "penalty_height": UnboundedContinuousTensorSpec(1),
            "reward_escape": UnboundedContinuousTensorSpec(1),
            "reward_stall": UnboundedContinuousTensorSpec(1),
            "stall": UnboundedContinuousTensorSpec(1),
            "stall_steps": UnboundedContinuousTensorSpec(1),
            "reward_vo": UnboundedContinuousTensorSpec(1),
            "vo_risk": UnboundedContinuousTensorSpec(1),
            "vo_warmup": UnboundedContinuousTensorSpec(1),
            "reward_action_vo": UnboundedContinuousTensorSpec(1),
            "action_vo_risk": UnboundedContinuousTensorSpec(1),
            "reward_static_vo": UnboundedContinuousTensorSpec(1),
            "static_vo_risk": UnboundedContinuousTensorSpec(1),
            "reward_safe_action": UnboundedContinuousTensorSpec(1),
            "safe_action_penalty": UnboundedContinuousTensorSpec(1),
            "safe_action_risk": UnboundedContinuousTensorSpec(1),
            "reach_goal": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "wall_collision": UnboundedContinuousTensorSpec(1),
            "below_bound": UnboundedContinuousTensorSpec(1),
            "above_bound": UnboundedContinuousTensorSpec(1),
            "horizontal_out_of_bound": UnboundedContinuousTensorSpec(1),
            "stuck": UnboundedContinuousTensorSpec(1),
            "stuck_active": UnboundedContinuousTensorSpec(1),
            "stuck_steps": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)

        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    def _get_static_clearance_map(self, clearance_radius: float):
        if self.terrain_occupancy is None:
            return None
        clearance_radius = max(float(clearance_radius), 0.0)
        pixel_radius = int(np.ceil(clearance_radius / self.terrain_occupancy_resolution))
        if pixel_radius not in self.terrain_clearance_cache:
            kernel_size = 2 * pixel_radius + 1
            occupancy = self.terrain_occupancy.unsqueeze(0).unsqueeze(0)
            expanded = F.max_pool2d(occupancy, kernel_size=kernel_size, stride=1, padding=pixel_radius)
            self.terrain_clearance_cache[pixel_radius] = expanded[0, 0] > 0.5
        return self.terrain_clearance_cache[pixel_radius]

    def _points_have_static_clearance(self, points_xy: torch.Tensor, clearance_radius: float):
        clearance_map = self._get_static_clearance_map(clearance_radius)
        if clearance_map is None:
            return torch.ones(points_xy.shape[0], dtype=torch.bool, device=self.device)

        half_size = 0.5 * self.terrain_occupancy_size
        local_xy = points_xy + half_size.unsqueeze(0)
        valid = (
            (local_xy[:, 0] >= 0.0)
            & (local_xy[:, 0] <= self.terrain_occupancy_size[0])
            & (local_xy[:, 1] >= 0.0)
            & (local_xy[:, 1] <= self.terrain_occupancy_size[1])
        )
        if not torch.any(valid).item():
            return torch.ones(points_xy.shape[0], dtype=torch.bool, device=self.device)

        map_x = clearance_map.shape[0] - 1
        map_y = clearance_map.shape[1] - 1
        grid_xy = torch.round(local_xy / self.terrain_occupancy_resolution).long()
        grid_xy[:, 0] = grid_xy[:, 0].clamp_(0, map_x)
        grid_xy[:, 1] = grid_xy[:, 1].clamp_(0, map_y)
        occupied = clearance_map[grid_xy[:, 0], grid_xy[:, 1]]
        return (~valid) | (~occupied)

    def _get_wall_collision_height_map(self, clearance_radius: float):
        if self.wall_height_map is None:
            return None
        clearance_radius = max(float(clearance_radius), 0.0)
        pixel_radius = int(np.ceil(clearance_radius / self.terrain_occupancy_resolution))
        if pixel_radius not in self.wall_collision_cache:
            kernel_size = 2 * pixel_radius + 1
            wall_heights = self.wall_height_map.unsqueeze(0).unsqueeze(0)
            expanded = F.max_pool2d(wall_heights, kernel_size=kernel_size, stride=1, padding=pixel_radius)
            self.wall_collision_cache[pixel_radius] = expanded[0, 0]
        return self.wall_collision_cache[pixel_radius]

    def _points_near_curriculum_wall(self, points_xyz: torch.Tensor, clearance_radius: float):
        wall_height_map = self._get_wall_collision_height_map(clearance_radius)
        if wall_height_map is None:
            return torch.zeros(points_xyz.shape[:-1], dtype=torch.bool, device=self.device)

        original_shape = points_xyz.shape[:-1]
        points_flat = points_xyz.reshape(-1, 3)
        half_size = 0.5 * self.terrain_occupancy_size
        local_xy = points_flat[:, :2] + half_size.unsqueeze(0)
        valid = (
            (local_xy[:, 0] >= 0.0)
            & (local_xy[:, 0] <= self.terrain_occupancy_size[0])
            & (local_xy[:, 1] >= 0.0)
            & (local_xy[:, 1] <= self.terrain_occupancy_size[1])
        )

        map_x = wall_height_map.shape[0] - 1
        map_y = wall_height_map.shape[1] - 1
        grid_xy = torch.round(local_xy / self.terrain_occupancy_resolution).long()
        grid_xy[:, 0] = grid_xy[:, 0].clamp_(0, map_x)
        grid_xy[:, 1] = grid_xy[:, 1].clamp_(0, map_y)

        wall_height = torch.zeros(points_flat.shape[0], dtype=torch.float32, device=self.device)
        if torch.any(valid).item():
            wall_height[valid] = wall_height_map[grid_xy[valid, 0], grid_xy[valid, 1]]
        near_wall_xy = wall_height > 0.0
        below_wall_top = points_flat[:, 2] <= (wall_height + float(clearance_radius))
        return (valid & near_wall_xy & below_wall_top).reshape(original_shape)

    def _observation_noise_active(self):
        if not self.obs_noise_enabled:
            return False
        return (not self.obs_noise_train_only) or bool(getattr(self, "training", True))

    def _apply_lidar_observation_noise(self, lidar_scan: torch.Tensor):
        if not self._observation_noise_active():
            return lidar_scan

        noisy_scan = lidar_scan.clone()
        if self.lidar_scale_std > 0.0:
            scale = 1.0 + torch.randn(
                (*noisy_scan.shape[:2], 1, 1),
                dtype=noisy_scan.dtype,
                device=noisy_scan.device,
            ) * self.lidar_scale_std
            noisy_scan = noisy_scan * scale
        if self.lidar_noise_std > 0.0:
            noisy_scan = noisy_scan + torch.randn_like(noisy_scan) * self.lidar_noise_std
        if self.lidar_dropout_prob > 0.0:
            dropout_mask = torch.rand_like(noisy_scan) < self.lidar_dropout_prob
            noisy_scan = torch.where(dropout_mask, torch.zeros_like(noisy_scan), noisy_scan)
        if self.lidar_quantization > 0.0:
            noisy_scan = torch.round(noisy_scan / self.lidar_quantization) * self.lidar_quantization
        return noisy_scan.clamp_(0.0, self.lidar_range)

    def _apply_observation_delay(self, obs):
        if not self._observation_noise_active() or self.observation_delay_steps <= 0:
            return obs

        if self.observation_delay_buffers is None:
            self.observation_delay_buffers = {
                key: value.unsqueeze(0).repeat(self.observation_delay_steps + 1, *([1] * value.dim())).clone()
                for key, value in obs.items()
            }
            return obs

        reset_mask = self.observation_delay_reset_mask
        if torch.any(reset_mask):
            for key, value in obs.items():
                reset_value = value[reset_mask].detach().unsqueeze(0).expand(
                    self.observation_delay_steps + 1, *value[reset_mask].shape
                )
                self.observation_delay_buffers[key][:, reset_mask] = reset_value
            self.observation_delay_reset_mask.zero_()

        delayed_obs = {
            key: self.observation_delay_buffers[key][0].clone()
            for key in obs.keys()
        }
        for key, value in obs.items():
            buffer = self.observation_delay_buffers[key]
            buffer[:-1] = buffer[1:].clone()
            buffer[-1] = value.detach()
        return delayed_obs

    def _apply_action_delay(self, actions: torch.Tensor):
        if not self._observation_noise_active() or self.action_delay_steps <= 0:
            return actions

        if self.action_delay_buffer is None or self.action_delay_buffer.shape[1:] != actions.shape:
            self.action_delay_buffer = actions.unsqueeze(0).repeat(
                self.action_delay_steps + 1, *([1] * actions.dim())
            ).clone()
            return actions

        reset_mask = self.action_delay_reset_mask
        if torch.any(reset_mask):
            reset_actions = actions[reset_mask].detach().unsqueeze(0).expand(
                self.action_delay_steps + 1, *actions[reset_mask].shape
            )
            self.action_delay_buffer[:, reset_mask] = reset_actions
            self.action_delay_reset_mask.zero_()

        delayed_actions = self.action_delay_buffer[0].clone()
        self.action_delay_buffer[:-1] = self.action_delay_buffer[1:].clone()
        self.action_delay_buffer[-1] = actions.detach()
        return delayed_actions

    def _apply_action_execution_noise(self, actions: torch.Tensor):
        if (
            not self._observation_noise_active()
            or (self.action_execution_noise_std <= 0.0 and self.action_execution_scale_std <= 0.0)
        ):
            return actions

        noisy_actions = actions.clone()
        if self.action_execution_scale_std > 0.0:
            scale_shape = noisy_actions.shape[:-1] + (1,)
            scale = 1.0 + torch.randn(scale_shape, device=noisy_actions.device, dtype=noisy_actions.dtype) * self.action_execution_scale_std
            noisy_actions = noisy_actions * scale
        if self.action_execution_noise_std > 0.0:
            noisy_actions = noisy_actions + torch.randn_like(noisy_actions) * self.action_execution_noise_std

        action_limit = float(getattr(self.cfg.algo.actor, "action_limit", 0.0))
        if action_limit > 0.0:
            noisy_actions = noisy_actions.clamp(-action_limit, action_limit)
        return noisy_actions

    def _apply_dynamic_obstacle_frame_dropout(self, dyn_obs_states: torch.Tensor):
        if not self._observation_noise_active() or self.dyn_obs_frame_dropout_prob <= 0.0:
            self.prev_dyn_obs_states = dyn_obs_states.detach().clone()
            return dyn_obs_states

        if self.prev_dyn_obs_states is None or self.prev_dyn_obs_states.shape != dyn_obs_states.shape:
            self.prev_dyn_obs_states = dyn_obs_states.detach().clone()
            return dyn_obs_states

        reset_mask = self.dyn_obs_frame_drop_reset_mask
        if torch.any(reset_mask):
            self.prev_dyn_obs_states[reset_mask] = dyn_obs_states[reset_mask].detach()
            self.dyn_obs_frame_drop_reset_mask.zero_()

        drop_shape = (dyn_obs_states.shape[0],) + (1,) * (dyn_obs_states.dim() - 1)
        drop_mask = torch.rand(drop_shape, device=dyn_obs_states.device) < self.dyn_obs_frame_dropout_prob
        observed_dyn_obs_states = torch.where(drop_mask, self.prev_dyn_obs_states, dyn_obs_states)
        self.prev_dyn_obs_states = observed_dyn_obs_states.detach().clone()
        return observed_dyn_obs_states

    def _apply_dynamic_obstacle_observation_noise(
        self,
        dyn_obs_rpos: torch.Tensor,
        dyn_obs_vel: torch.Tensor,
        dyn_obs_size: torch.Tensor,
        dyn_obs_range_mask: torch.Tensor,
    ):
        if not self._observation_noise_active():
            return dyn_obs_rpos, dyn_obs_vel, dyn_obs_size, dyn_obs_range_mask

        noisy_rpos = dyn_obs_rpos.clone()
        noisy_vel = dyn_obs_vel.clone()
        noisy_size = dyn_obs_size.clone()
        noisy_range_mask = dyn_obs_range_mask.clone()
        tall_obstacle = dyn_obs_size[..., 2] > self.max_obs_3d_height
        three_d_noise_mask = (~tall_obstacle).unsqueeze(-1).to(dtype=noisy_rpos.dtype)

        if self.dyn_obs_pos_noise_std > 0.0:
            pos_noise = torch.randn_like(noisy_rpos) * self.dyn_obs_pos_noise_std
            pos_noise[..., 2:] = pos_noise[..., 2:] * three_d_noise_mask
            noisy_rpos = noisy_rpos + pos_noise
        if self.dyn_obs_vel_noise_std > 0.0:
            vel_noise = torch.randn_like(noisy_vel) * self.dyn_obs_vel_noise_std
            vel_noise[..., 2:] = vel_noise[..., 2:] * three_d_noise_mask
            noisy_vel = noisy_vel + vel_noise
        if self.dyn_obs_size_noise_std > 0.0:
            noisy_size = (noisy_size + torch.randn_like(noisy_size) * self.dyn_obs_size_noise_std).clamp_min(0.05)
        if self.dyn_obs_dropout_prob > 0.0:
            dropout_mask = torch.rand_like(noisy_range_mask.float()) < self.dyn_obs_dropout_prob
            noisy_range_mask = noisy_range_mask | dropout_mask

        return noisy_rpos, noisy_vel, noisy_size, noisy_range_mask

    @staticmethod
    def _normalize_task_mode(mode: str):
        aliases = {
            "opposite_crossing_eval": "opposite_crossing",
            "random_crossing_eval": "opposite_crossing",
            "random_crossing": "opposite_crossing",
            "random": "opposite_crossing",
            "wall_crossing_eval": "wall_crossing",
        }
        normalized = aliases.get(str(mode), str(mode))
        if normalized not in ("opposite_crossing", "wall_crossing"):
            raise ValueError(
                f"Unknown task style={mode}. Expected 'opposite_crossing' or 'wall_crossing'."
            )
        return normalized

    def _sample_opposite_boundary_candidates(self, count: int):
        boundary = self.task_boundary_coordinate
        tangent = (2.0 * torch.rand(count, device=self.device) - 1.0) * self.task_tangent_limit
        jitter = (2.0 * torch.rand(count, device=self.device) - 1.0) * self.task_target_jitter
        target_tangent = (-tangent + jitter).clamp(-self.task_tangent_limit, self.task_tangent_limit)
        side = torch.randint(0, 4, (count,), device=self.device)

        start = torch.zeros(count, 1, 3, dtype=torch.float, device=self.device)
        target = torch.zeros_like(start)
        top = side == 0
        bottom = side == 1
        right = side == 2
        left = side == 3

        start[top, 0, 0] = tangent[top]
        start[top, 0, 1] = boundary
        target[top, 0, 0] = target_tangent[top]
        target[top, 0, 1] = -boundary

        start[bottom, 0, 0] = tangent[bottom]
        start[bottom, 0, 1] = -boundary
        target[bottom, 0, 0] = target_tangent[bottom]
        target[bottom, 0, 1] = boundary

        start[right, 0, 0] = boundary
        start[right, 0, 1] = tangent[right]
        target[right, 0, 0] = -boundary
        target[right, 0, 1] = target_tangent[right]

        start[left, 0, 0] = -boundary
        start[left, 0, 1] = tangent[left]
        target[left, 0, 0] = boundary
        target[left, 0, 1] = target_tangent[left]

        start[:, 0, 2] = 0.5 + 2.0 * torch.rand(count, device=self.device)
        target[:, 0, 2] = 0.5 + 2.0 * torch.rand(count, device=self.device)
        return start, target

    def _points_have_dynamic_clearance(self, points_xyz: torch.Tensor, clearance_radius: float):
        if (
            not hasattr(self, "dyn_obs_state")
            or not hasattr(self, "dyn_obs_size")
            or self.dyn_obs_state.shape[0] == 0
        ):
            return torch.ones(points_xyz.shape[0], dtype=torch.bool, device=self.device)

        obstacle_pos = self.dyn_obs_state[:, :3]
        obstacle_size = self.dyn_obs_size
        delta = points_xyz.unsqueeze(1) - obstacle_pos.unsqueeze(0)
        horizontal_distance = delta[..., :2].norm(dim=-1)
        obstacle_radius = 0.5 * obstacle_size[..., :2].norm(dim=-1)
        horizontal_overlap = horizontal_distance <= (obstacle_radius.unsqueeze(0) + clearance_radius)
        vertical_overlap = delta[..., 2].abs() <= (
            0.5 * obstacle_size[:, 2].unsqueeze(0) + clearance_radius
        )
        return ~torch.any(horizontal_overlap & vertical_overlap, dim=1)

    def _segments_intersect_wall_influence(self, start_xy: torch.Tensor, target_xy: torch.Tensor):
        if self.wall_occupancy is None or not torch.any(self.wall_occupancy > 0.5).item():
            return torch.zeros(start_xy.shape[0], dtype=torch.bool, device=self.device)

        pixel_radius = int(np.ceil(self.wall_influence_radius / self.terrain_occupancy_resolution))
        if pixel_radius not in self.wall_influence_cache:
            kernel_size = 2 * pixel_radius + 1
            occupancy = self.wall_occupancy.unsqueeze(0).unsqueeze(0)
            expanded = F.max_pool2d(occupancy, kernel_size=kernel_size, stride=1, padding=pixel_radius)
            self.wall_influence_cache[pixel_radius] = expanded[0, 0] > 0.5
        influence_map = self.wall_influence_cache[pixel_radius]

        max_line_length = 2.0 * np.sqrt(
            self.task_boundary_coordinate**2 + self.task_tangent_limit**2
        )
        sample_count = max(2, int(np.ceil(max_line_length / self.wall_line_sample_spacing)) + 1)
        alpha = torch.linspace(0.0, 1.0, sample_count, device=self.device)
        points = start_xy.unsqueeze(1) + alpha.view(1, -1, 1) * (
            target_xy - start_xy
        ).unsqueeze(1)

        half_size = 0.5 * self.terrain_occupancy_size
        local_xy = points + half_size.view(1, 1, 2)
        valid = (
            (local_xy[..., 0] >= 0.0)
            & (local_xy[..., 0] <= self.terrain_occupancy_size[0])
            & (local_xy[..., 1] >= 0.0)
            & (local_xy[..., 1] <= self.terrain_occupancy_size[1])
        )
        grid_xy = torch.round(local_xy / self.terrain_occupancy_resolution).long()
        grid_xy[..., 0].clamp_(0, influence_map.shape[0] - 1)
        grid_xy[..., 1].clamp_(0, influence_map.shape[1] - 1)
        intersects = influence_map[grid_xy[..., 0], grid_xy[..., 1]] & valid
        return torch.any(intersects, dim=1)

    def _sample_task_pairs(self, count: int, mode: str):
        mode = self._normalize_task_mode(mode)
        require_wall = mode == "wall_crossing"
        if require_wall and (
            self.wall_occupancy is None or not torch.any(self.wall_occupancy > 0.5).item()
        ):
            return self._sample_task_pairs(count, "opposite_crossing")

        starts = torch.zeros(count, 1, 3, dtype=torch.float, device=self.device)
        targets = torch.zeros_like(starts)
        remaining = torch.arange(count, device=self.device)
        fallback = None

        for _ in range(self.position_sample_attempts):
            if remaining.numel() == 0:
                break
            candidate_start, candidate_target = self._sample_opposite_boundary_candidates(
                int(remaining.numel())
            )
            fallback = (candidate_start, candidate_target)
            valid = self._points_have_static_clearance(
                candidate_start[:, 0, :2], self.spawn_clearance_radius
            )
            valid &= self._points_have_static_clearance(
                candidate_target[:, 0, :2], self.target_clearance_radius
            )
            valid &= self._points_have_dynamic_clearance(
                candidate_start[:, 0, :], self.task_dynamic_clearance
            )
            valid &= self._points_have_dynamic_clearance(
                candidate_target[:, 0, :], self.task_dynamic_clearance
            )
            if require_wall:
                valid &= self._segments_intersect_wall_influence(
                    candidate_start[:, 0, :2], candidate_target[:, 0, :2]
                )
            if not torch.any(valid).item():
                continue
            accepted = remaining[valid]
            starts[accepted] = candidate_start[valid]
            targets[accepted] = candidate_target[valid]
            remaining = remaining[~valid]

        if remaining.numel() > 0:
            if require_wall:
                fallback_start, fallback_target = self._sample_task_pairs(
                    int(remaining.numel()), "opposite_crossing"
                )
            else:
                if fallback is None or fallback[0].shape[0] != remaining.numel():
                    fallback = self._sample_opposite_boundary_candidates(int(remaining.numel()))
                fallback_start, fallback_target = fallback
            starts[remaining] = fallback_start
            targets[remaining] = fallback_target
        return starts, targets

    def set_eval_task_mode(self, mode: str):
        self.eval_task_mode = "standard" if mode == "standard" else self._normalize_task_mode(mode)

    def _set_standard_eval_task(self, env_ids: torch.Tensor):
        # Deterministic top-to-bottom crossing with drones spread along the x axis.
        pos = torch.zeros(len(env_ids), 1, 3, dtype=torch.float, device=self.device)
        pos[:, 0, 0] = (env_ids.float() / self.num_envs - 0.5) * 32.0
        pos[:, 0, 1] = self.task_boundary_coordinate
        pos[:, 0, 2] = 2.0

        target_x = torch.linspace(-0.5, 0.5, self.num_envs, dtype=torch.float, device=self.device) * 32.0
        self.target_pos[env_ids, 0, 0] = target_x[env_ids.long()]
        self.target_pos[env_ids, 0, 1] = -self.task_boundary_coordinate
        self.target_pos[env_ids, 0, 2] = 2.0
        return pos

    
    def reset_target(self, env_ids: torch.Tensor):
        if (self.training):
            _, targets = self._sample_task_pairs(env_ids.size(0), self.train_task_mode)
            self.target_pos[env_ids] = targets

            # self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            # self.target_pos[:, 0, 1] = 24.
            # self.target_pos[:, 0, 2] = 2.    
        else:
            self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            self.target_pos[:, 0, 1] = -self.task_boundary_coordinate
            self.target_pos[:, 0, 2] = 2.            


    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        if (self.training):
            pos, targets = self._sample_task_pairs(env_ids.size(0), self.train_task_mode)
            self.target_pos[env_ids] = targets

            # pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            # pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            # pos[:, 0, 1] = -24.
            # pos[:, 0, 2] = 2.
        else:
            if self.eval_task_mode != "standard":
                pos, targets = self._sample_task_pairs(env_ids.size(0), self.eval_task_mode)
                self.target_pos[env_ids] = targets
            else:
                pos = self._set_standard_eval_task(env_ids)
        
        # Coordinate change: after reset, the drone's target direction should be changed
        self.target_dir[env_ids] = self.target_pos[env_ids] - pos

        # Coordinate change: after reset, the drone's facing direction should face the current goal
        rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        diff = self.target_pos[env_ids] - pos
        facing_yaw = torch.atan2(diff[..., 1], diff[..., 0])
        rpy[..., 2] = facing_yaw

        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.prev_drone_vel_w[env_ids] = 0.
        init_goal_distance = (self.target_pos[env_ids] - pos).norm(dim=-1)
        self.prev_goal_distance[env_ids] = init_goal_distance
        self.prev_front_clearance[env_ids] = self.lidar_range
        self.stuck_counter[env_ids] = 0
        self.escape_armed[env_ids] = False
        self.escape_age[env_ids] = 0
        self.escape_cooldown[env_ids] = 0
        self.escape_anchor_pos[env_ids] = pos
        self.escape_anchor_goal_distance[env_ids] = init_goal_distance
        self.escape_anchor_clearance[env_ids] = self.lidar_range
        self.stall_counter[env_ids] = 0
        self.stall_anchor_pos[env_ids] = pos
        self.height_range[env_ids, 0, 0] = torch.min(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])
        self.height_range[env_ids, 0, 1] = torch.max(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])
        self.observation_delay_reset_mask[env_ids] = True
        self.action_delay_reset_mask[env_ids] = True
        self.dyn_obs_frame_drop_reset_mask[env_ids] = True

        self.stats[env_ids] = 0.  
        
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.policy_action_w = actions.detach().clone()
        actions = self._apply_action_delay(actions)
        actions = self._apply_action_execution_noise(actions)
        self.executed_action_w = actions.detach().clone()
        if self.training:
            self.vo_step_count += 1
        self.drone.apply_action(actions) 

    def _post_sim_step(self, tensordict: TensorDictBase):
        if (self.cfg.env_dyn.num_obstacles != 0):
            self.move_dynamic_obstacle()
        self.lidar.update(self.dt)
    
    # get current states/observation
    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state(env_frame=False) # (world_pos, orientation (quat), world_vel_and_angular, heading, up, 4motorsthrust)
        self.info["drone_state"][:] = self.root_state[..., :13] # info is for controller

        # >>>>>>>>>>>>The relevant code starts from here<<<<<<<<<<<<
        # -----------Network Input I: LiDAR range data--------------
        self.lidar_scan_clean = self.lidar_range - (
            (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
            .norm(dim=-1)
            .clamp_max(self.lidar_range)
            .reshape(self.num_envs, 1, *self.lidar_resolution)
        ) # lidar scan store the data that is range - distance and it is in lidar's local frame
        self.lidar_scan = self._apply_lidar_observation_noise(self.lidar_scan_clean)

        # Optional render for LiDAR
        if self._should_render(0):
            self.debug_draw.clear()
            x = self.lidar.data.pos_w[0]
            # set_camera_view(
            #     eye=x.cpu() + torch.as_tensor(self.cfg.viewer.eye),
            #     target=x.cpu() + torch.as_tensor(self.cfg.viewer.lookat)                        
            # )
            v = (self.lidar.data.ray_hits_w[0] - x).reshape(*self.lidar_resolution, 3)
            # self.debug_draw.vector(x.expand_as(v[:, 0]), v[:, 0])
            # self.debug_draw.vector(x.expand_as(v[:, -1]), v[:, -1])
            self.debug_draw.vector(x.expand_as(v[:, 0])[0], v[0, 0])

        # ---------Network Input II: Drone's internal states---------
        # a. distance info in horizontal and vertical plane
        rpos = self.target_pos - self.root_state[..., :3]        
        distance = rpos.norm(dim=-1, keepdim=True) # start to goal distance
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)
        distance_z = rpos[..., 2].unsqueeze(-1)
        
        
        # b. Unit direction to the goal in the episode-fixed navigation frame.
        # The frame orientation is initialized from start to goal at reset.
        target_dir_2d = self.target_dir.clone()
        target_dir_2d[..., 2] = 0

        rpos_clipped = rpos / distance.clamp(1e-6) # unit vector: start to goal direction
        rpos_clipped_g = vec_to_new_frame(rpos_clipped, target_dir_2d) # express in the goal coodinate
        stuck_dir_2d = rpos.clone()
        stuck_dir_2d[..., 2] = 0
        stuck_dir_2d = torch.where(stuck_dir_2d.norm(dim=-1, keepdim=True) > 1e-6, stuck_dir_2d, target_dir_2d)

        lidar_hit_rpos_w = self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1)
        lidar_hit_rpos_g = vec_to_new_frame(lidar_hit_rpos_w, stuck_dir_2d)
        static_front_mask = self._in_stuck_front_region(lidar_hit_rpos_g)
        static_front_obstacle = static_front_mask.any(dim=1, keepdim=True)
        lidar_hit_distance = lidar_hit_rpos_w.norm(dim=-1)
        front_clearance_values = torch.where(
            static_front_mask,
            lidar_hit_distance,
            torch.full_like(lidar_hit_distance, self.lidar_range),
        )
        front_clearance = front_clearance_values.min(dim=1, keepdim=True).values.clamp(max=self.lidar_range)
        static_vo_hit_mask = (lidar_hit_distance < (self.lidar_range - 1e-3)) & (
            lidar_hit_rpos_w[..., 2] >= self.vo_static_vertical_min
        )
        static_vo_range_mask = ~static_vo_hit_mask
        static_vo_rpos = lidar_hit_rpos_w
        static_vo_vel = torch.zeros_like(static_vo_rpos)
        static_vo_size = torch.full_like(static_vo_rpos, 2.0 * self.vo_static_radius)
        
        # c. velocity in the goal frame
        vel_w = self.root_state[..., 7:10] # world vel
        vel_g = vec_to_new_frame(vel_w, target_dir_2d)   # coordinate change for velocity
        policy_action_w = self._format_vo_velocity(getattr(self, "policy_action_w", vel_w))

        # final drone's internal states
        drone_state = torch.cat([rpos_clipped_g, distance_2d, distance_z, vel_g], dim=-1).squeeze(1)

        if (self.cfg.env_dyn.num_obstacles != 0):
            # ---------Network Input III: Dynamic obstacle states--------
            # ------------------------------------------------------------
            # a. Closest N obstacles relative position in the goal frame 
            # Find the N closest and within range obstacles for each drone
            dyn_obs_pos_expanded = self.dyn_obs_state[..., :3].unsqueeze(0).repeat(self.num_envs, 1, 1)
            dyn_obs_rpos_expanded = dyn_obs_pos_expanded[..., :3] - self.root_state[..., :3] 
            dyn_obs_rpos_expanded[:, int(self.dyn_obs_state.size(0)/2):, 2] = 0.
            dyn_obs_rpos_expanded_g_stuck = vec_to_new_frame(dyn_obs_rpos_expanded, stuck_dir_2d)
            dynamic_front_obstacle = self._in_stuck_front_region(
                dyn_obs_rpos_expanded_g_stuck,
                lateral_inflation=self.dyn_obs_size[:, 0].unsqueeze(0) / 2.,
                vertical_inflation=self.dyn_obs_size[:, 2].unsqueeze(0) / 2.,
            ).any(dim=1, keepdim=True)
            dyn_obs_distance_2d = torch.norm(dyn_obs_rpos_expanded[..., :2], dim=2)  # Shape: (1000, 40). calculate 2d distance to each obstacle for all drones
            _, closest_dyn_obs_idx = torch.topk(dyn_obs_distance_2d, self.cfg.algo.feature_extractor.dyn_obs_num, dim=1, largest=False) # pick top N closest obstacle index for RL observation
            dyn_obs_range_mask = dyn_obs_distance_2d.gather(1, closest_dyn_obs_idx) > self.lidar_range

            # relative distance of obstacles in the goal frame
            closest_dyn_obs_rpos = torch.gather(dyn_obs_rpos_expanded, 1, closest_dyn_obs_idx.unsqueeze(-1).expand(-1, -1, 3))

            # b. Velocity in the goal frame for the dynamic obstacles
            closest_dyn_obs_vel = self.dyn_obs_vel[closest_dyn_obs_idx]

            # c. Size of dynamic obstacles in category
            closest_dyn_obs_size = self.dyn_obs_size[closest_dyn_obs_idx] # the acutal size
            closest_dyn_obs_width = closest_dyn_obs_size[..., 0].unsqueeze(-1)
            closest_dyn_obs_height = closest_dyn_obs_size[..., 2].unsqueeze(-1)

            observed_dyn_obs_rpos, observed_dyn_obs_vel, observed_dyn_obs_size, observed_dyn_obs_range_mask = (
                self._apply_dynamic_obstacle_observation_noise(
                    closest_dyn_obs_rpos,
                    closest_dyn_obs_vel,
                    closest_dyn_obs_size,
                    dyn_obs_range_mask,
                )
            )
            observed_dyn_obs_rpos_g = vec_to_new_frame(observed_dyn_obs_rpos, target_dir_2d)
            observed_dyn_obs_rpos_g[observed_dyn_obs_range_mask] = 0. # exclude out of range or dropped obstacles
            observed_dyn_obs_distance = observed_dyn_obs_rpos.norm(dim=-1, keepdim=True)
            observed_dyn_obs_distance_2d = observed_dyn_obs_rpos_g[..., :2].norm(dim=-1, keepdim=True)
            observed_dyn_obs_distance_z = observed_dyn_obs_rpos_g[..., 2].unsqueeze(-1)
            observed_dyn_obs_rpos_gn = observed_dyn_obs_rpos_g / observed_dyn_obs_distance.clamp(1e-6)

            observed_dyn_obs_vel[observed_dyn_obs_range_mask] = 0.
            observed_dyn_obs_vel_g = vec_to_new_frame(observed_dyn_obs_vel, target_dir_2d)

            observed_dyn_obs_width = observed_dyn_obs_size[..., 0].unsqueeze(-1)
            closest_dyn_obs_width_category = (observed_dyn_obs_width / self.dyn_obs_width_res - 1.).clamp(0., 3.) # convert to category: [0, 1, 2, 3]
            closest_dyn_obs_width_category[observed_dyn_obs_range_mask] = 0.

            observed_dyn_obs_height = observed_dyn_obs_size[..., 2].unsqueeze(-1)
            closest_dyn_obs_height_category = torch.where(
                observed_dyn_obs_height > self.max_obs_3d_height,
                torch.zeros_like(observed_dyn_obs_height),
                observed_dyn_obs_height,
            )
            closest_dyn_obs_height_category[observed_dyn_obs_range_mask] = 0.

            # concatenate all for dynamic obstacles
            # dyn_obs_states = torch.cat([closest_dyn_obs_rpos_g, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)
            dyn_obs_states = torch.cat([observed_dyn_obs_rpos_gn, observed_dyn_obs_distance_2d, observed_dyn_obs_distance_z, observed_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)
            dyn_obs_states = self._apply_dynamic_obstacle_frame_dropout(dyn_obs_states)

            closest_dyn_obs_vel[dyn_obs_range_mask] = 0.

            # check dynamic obstacle collision for later reward
            closest_dyn_obs_distance_2d_collsion = closest_dyn_obs_rpos[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d_collsion[dyn_obs_range_mask] = float('inf')
            closest_dyn_obs_distance_zn_collision = closest_dyn_obs_rpos[..., 2].unsqueeze(-1).norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_zn_collision[dyn_obs_range_mask] = float('inf')
            dynamic_collision_2d = closest_dyn_obs_distance_2d_collsion <= (closest_dyn_obs_width/2. + 0.3)
            dynamic_collision_z = closest_dyn_obs_distance_zn_collision <= (closest_dyn_obs_height/2. + 0.3)
            dynamic_collision_each = dynamic_collision_2d & dynamic_collision_z
            dynamic_collision = torch.any(dynamic_collision_each, dim=1)

            # distance to dynamic obstacle for reward calculation (not 100% correct in math but should be good enough for approximation)
            closest_dyn_obs_distance_reward = closest_dyn_obs_rpos.norm(dim=-1) - closest_dyn_obs_size[..., 0]/2. # for those 2D obstacle, z distance will not be considered
            closest_dyn_obs_distance_reward[dyn_obs_range_mask] = self.cfg.sensor.lidar_range

            # Keep reward candidates aligned with the observed dynamic obstacles, then
            # re-rank only within that set using the VO metric.
            closest_dyn_obs_distance_3d = closest_dyn_obs_rpos.norm(dim=-1)
            reward_dyn_obs_metric = torch.norm(
                closest_dyn_obs_rpos / self._compute_vo_scale(closest_dyn_obs_size),
                dim=2,
            )
            reward_dyn_obs_metric = reward_dyn_obs_metric.masked_fill(
                dyn_obs_range_mask,
                float("inf"),
            )
            reward_dyn_obs_metric = reward_dyn_obs_metric.masked_fill(
                closest_dyn_obs_distance_3d > self.vo_reward_range,
                float("inf"),
            )
            reward_topk = min(self.vo_reward_topk, reward_dyn_obs_metric.size(1))
            _, reward_dyn_obs_idx = torch.topk(reward_dyn_obs_metric, reward_topk, dim=1, largest=False)
            reward_dyn_obs_distance_3d = torch.gather(closest_dyn_obs_distance_3d, 1, reward_dyn_obs_idx)
            reward_dyn_obs_range_mask = torch.gather(dyn_obs_range_mask, 1, reward_dyn_obs_idx) | (reward_dyn_obs_distance_3d > self.vo_reward_range)
            reward_dyn_obs_rpos = torch.gather(closest_dyn_obs_rpos, 1, reward_dyn_obs_idx.unsqueeze(-1).expand(-1, -1, 3))
            reward_dyn_obs_vel = torch.gather(closest_dyn_obs_vel, 1, reward_dyn_obs_idx.unsqueeze(-1).expand(-1, -1, 3))
            reward_dyn_obs_size = torch.gather(closest_dyn_obs_size, 1, reward_dyn_obs_idx.unsqueeze(-1).expand(-1, -1, 3))
            reward_dyn_obs_rpos, reward_dyn_obs_vel, reward_dyn_obs_size, reward_dyn_obs_range_mask = (
                self._expand_dynamic_obstacles_for_vo(
                    reward_dyn_obs_rpos,
                    reward_dyn_obs_vel,
                    reward_dyn_obs_size,
                    reward_dyn_obs_range_mask,
                    vel_w,
                )
            )
            reward_vo, vo_risk, vo_warmup, _ = self._compute_vo_reward(
                reward_dyn_obs_rpos,
                reward_dyn_obs_vel,
                reward_dyn_obs_size,
                reward_dyn_obs_range_mask,
                vel_w,
            )
            reward_action_vo, action_vo_risk, _, _ = self._compute_vo_reward(
                reward_dyn_obs_rpos,
                reward_dyn_obs_vel,
                reward_dyn_obs_size,
                reward_dyn_obs_range_mask,
                policy_action_w,
                weight=self.vo_action_weight,
            )
            
        else:
            dyn_obs_states = torch.zeros(self.num_envs, 1, self.cfg.algo.feature_extractor.dyn_obs_num, 10, device=self.cfg.device)
            dyn_obs_states = self._apply_dynamic_obstacle_frame_dropout(dyn_obs_states)
            dynamic_collision = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.cfg.device)
            dynamic_front_obstacle = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.cfg.device)
            closest_dyn_obs_distance_reward = torch.full(
                (self.num_envs, self.cfg.algo.feature_extractor.dyn_obs_num),
                self.lidar_range,
                device=self.cfg.device,
            )
            reward_vo = torch.zeros(self.num_envs, 1, device=self.cfg.device)
            vo_risk = torch.zeros(self.num_envs, 1, device=self.cfg.device)
            vo_warmup = torch.zeros(self.num_envs, 1, device=self.cfg.device)
            reward_action_vo = torch.zeros(self.num_envs, 1, device=self.cfg.device)
            action_vo_risk = torch.zeros(self.num_envs, 1, device=self.cfg.device)
            reward_dyn_obs_rpos = torch.empty(self.num_envs, 0, 3, device=self.cfg.device)
            reward_dyn_obs_vel = torch.empty(self.num_envs, 0, 3, device=self.cfg.device)
            reward_dyn_obs_size = torch.empty(self.num_envs, 0, 3, device=self.cfg.device)
            reward_dyn_obs_range_mask = torch.empty(self.num_envs, 0, dtype=torch.bool, device=self.cfg.device)

        reward_static_vo, static_vo_risk, _, _ = self._compute_vo_reward(
            static_vo_rpos,
            static_vo_vel,
            static_vo_size,
            static_vo_range_mask,
            policy_action_w,
            weight=self.vo_static_weight,
        )
        safe_obs_rpos = torch.cat([reward_dyn_obs_rpos, static_vo_rpos], dim=1)
        safe_obs_vel = torch.cat([reward_dyn_obs_vel, static_vo_vel], dim=1)
        safe_obs_size = torch.cat([reward_dyn_obs_size, static_vo_size], dim=1)
        safe_obs_range_mask = torch.cat([reward_dyn_obs_range_mask, static_vo_range_mask], dim=1)
        safe_action_penalty, safe_action_risk = self._compute_safe_action_penalty(
            policy_action_w,
            safe_obs_rpos,
            safe_obs_vel,
            safe_obs_size,
            safe_obs_range_mask,
        )
        reward_safe_action = -self.vo_safe_action_weight * self._vo_warmup_like(safe_action_penalty) * safe_action_penalty

        goal_distance = distance.squeeze(-1)
        reach_goal = goal_distance <= self.goal_radius
        time_limit = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        goal_progress = self.prev_goal_distance - goal_distance
        front_obstacle = static_front_obstacle | dynamic_front_obstacle
        reward_goal_progress = self._compute_goal_progress_reward(
            goal_progress,
            reach_goal,
            time_limit,
        )
        reward_stall, stall_active = self._compute_stall_reward(
            self.root_state[..., :3],
            self.drone.vel_w[..., :3],
            goal_progress,
            reach_goal,
        )
        blocked_low_progress = (goal_progress <= self.stuck_progress_eps) & front_obstacle
        previous_stuck_counter = self.stuck_counter.clone()
        self.stuck_counter = torch.where(
            blocked_low_progress,
            self.stuck_counter + 1,
            torch.zeros_like(self.stuck_counter),
        )
        reward_escape = self._compute_escape_reward(
            previous_stuck_counter,
            self.root_state[..., :3],
            goal_distance,
            front_clearance,
            blocked_low_progress,
            reach_goal,
            time_limit,
        )
        stuck = self.stuck_counter >= self.stuck_window
        front_clearance_gain = front_clearance - self.prev_front_clearance
        self.prev_goal_distance = goal_distance.clone()
            
        # -----------------Network Input Final--------------
        obs = {
            "state": drone_state,
            "lidar": self.lidar_scan,
            "direction": target_dir_2d,
            "dynamic_obstacle": dyn_obs_states
        }
        obs = self._apply_observation_delay(obs)
####

        # -----------------Reward Calculation-----------------
        # a. penalize static obstacles only when they get too close.
        dist_static = self.lidar_range - self.lidar_scan_clean
        safe_margin = 1.1
        penalty_safety_static = torch.relu(safe_margin - dist_static) / safe_margin
        penalty_safety_static = penalty_safety_static.mean(dim=(2, 3))
        

        # b. penalize dynamic obstacles only when they enter the same safety margin.
        penalty_safety_dynamic = torch.relu(safe_margin - closest_dyn_obs_distance_reward) / safe_margin
        penalty_safety_dynamic = penalty_safety_dynamic.mean(dim=-1, keepdim=True)

        # c. reward forward progress only in the horizontal plane.
        goal_direction_xy = rpos[..., :2] / distance_2d.clamp_min(1e-6)
        reward_vel = (self.drone.vel_w[..., :2] * goal_direction_xy).sum(-1)
        
        # d. smoothness reward for action smoothness
        penalty_smooth = (self.drone.vel_w[..., :3] - self.prev_drone_vel_w).norm(dim=-1)
        
        # e. softly keep altitude close to the goal altitude.
        target_z = self.target_pos[..., 2]
        z_error = self.drone.pos[..., 2] - target_z
        penalty_height = torch.relu(z_error.abs() - 0.4).square()


        # f. Collision condition with its penalty
        static_collision = einops.reduce(self.lidar_scan_clean, "n 1 w h -> n 1", "max") >  (self.lidar_range - 0.3) # 0.3 collision radius
        wall_collision = static_collision & self._points_near_curriculum_wall(self.root_state[..., :3], clearance_radius=0.3)
        collision = static_collision | dynamic_collision
        below_bound = self.drone.pos[..., 2] < 0.2
        above_bound = self.drone.pos[..., 2] > 4.
        horizontal_out_of_bound = torch.any(
            self.drone.pos[..., :2].abs() > self.horizontal_limit, dim=-1
        )
        
        # Final reward calculation
        if (self.cfg.env_dyn.num_obstacles != 0):
            self.reward = reward_vel*0 + 0.1 - penalty_safety_static * 0.6 - penalty_safety_dynamic * 0.6 - penalty_smooth * 0.1 - penalty_height * 0.5
        else:
            self.reward = reward_vel*0 + 0.1 - penalty_safety_static * 0.6 - penalty_smooth * 0.1 - penalty_height * 0.5
        self.reward = (
            self.reward
            + reward_goal_progress
            + reward_escape
            + reward_stall
            + reward_vo
            + reward_action_vo*0
            + reward_static_vo*0
            + reward_safe_action*0
        )

        # Terminal penalties make failure modes explicitly costly.
        self.reward[collision] -= 45.0
        self.reward[below_bound] -= 20.0
        self.reward[above_bound] -= 20.0
        self.reward[horizontal_out_of_bound] -= 20.0

        # Terminate Conditions
        self.terminated = reach_goal | below_bound | above_bound | horizontal_out_of_bound | collision
        self.truncated = time_limit # progress buf is to track the step number

        # update previous velocity for smoothness calculation in the next ieteration
        self.prev_drone_vel_w = self.drone.vel_w[..., :3].clone()
        self.prev_front_clearance = front_clearance.clone()

        # # -----------------Training Stats-----------------
        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["goal_distance"] = goal_distance
        self.stats["goal_progress"] = goal_progress
        self.stats["front_obstacle"] = front_obstacle.float()
        self.stats["front_clearance"] = front_clearance
        self.stats["front_clearance_gain"] = front_clearance_gain
        self.stats["reward_goal_progress"] = reward_goal_progress
        self.stats["penalty_safety_static"] = penalty_safety_static
        self.stats["penalty_safety_dynamic"] = penalty_safety_dynamic
        self.stats["reward_vel"] = reward_vel
        self.stats["penalty_height"] = penalty_height
        self.stats["reward_escape"] = reward_escape
        self.stats["reward_stall"] = reward_stall
        self.stats["stall"] = torch.maximum(self.stats["stall"], stall_active.float())
        self.stats["stall_steps"] += stall_active.float()
        self.stats["reward_vo"] = reward_vo
        self.stats["vo_risk"] = vo_risk
        self.stats["vo_warmup"] = vo_warmup
        self.stats["reward_action_vo"] = reward_action_vo
        self.stats["action_vo_risk"] = action_vo_risk
        self.stats["reward_static_vo"] = reward_static_vo
        self.stats["static_vo_risk"] = static_vo_risk
        self.stats["reward_safe_action"] = reward_safe_action
        self.stats["safe_action_penalty"] = safe_action_penalty
        self.stats["safe_action_risk"] = safe_action_risk
        self.stats["reach_goal"] = reach_goal.float()
        self.stats["collision"] = collision.float()
        self.stats["wall_collision"] = wall_collision.float()
        self.stats["below_bound"] = below_bound.float()
        self.stats["above_bound"] = above_bound.float()
        self.stats["horizontal_out_of_bound"] = horizontal_out_of_bound.float()
        self.stats["stuck"] = torch.maximum(self.stats["stuck"], stuck.float())
        self.stats["stuck_active"] = stuck.float()
        self.stats["stuck_steps"] += stuck.float()
        self.stats["truncated"] = self.truncated.float()

        return TensorDict({
            "agents": TensorDict(
                {
                    "observation": obs,
                }, 
                [self.num_envs]
            ),
            "stats": self.stats.clone(),
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        reward = self.reward
        terminated = self.terminated
        truncated = self.truncated
        return TensorDict(
            {
                "agents": {
                    "reward": reward
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
