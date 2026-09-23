import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, IndependentNormal, Actor, GAE, make_batch, IndependentBeta, BetaActor, vec_to_world

class PPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.n_agents, self.action_dim = action_spec.shape

        
        # Feature extractor for LiDAR
        feature_extractor_network = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128),
        ).to(self.device)
        
        # Dynamic obstacle information extractor
        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            make_mlp([128, 64])
        ).to(self.device)

        # Feature extractor
        self.feature_extractor = TensorDictSequential(
            TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], ["_cnn_feature"]),
            TensorDictModule(dynamic_obstacle_network, [("agents", "observation", "dynamic_obstacle")], ["_dynamic_obstacle_feature"]),
            CatTensors(["_cnn_feature", ("agents", "observation", "state"), "_dynamic_obstacle_feature"], "_feature", del_keys=False), 
            TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
        ).to(self.device)

        auxiliary_cfg = cfg.feature_extractor.get("auxiliary", {})
        self.auxiliary_enabled = bool(auxiliary_cfg.get("enabled", False))
        self.auxiliary_loss_weight = float(auxiliary_cfg.get("loss_weight", 0.0))
        self.auxiliary_future_horizons = []
        for horizon in auxiliary_cfg.get("future_horizons", [1]):
            horizon = int(horizon)
            if horizon >= 1 and horizon not in self.auxiliary_future_horizons:
                self.auxiliary_future_horizons.append(horizon)
        if self.auxiliary_enabled and not self.auxiliary_future_horizons:
            self.auxiliary_future_horizons = [1]
        self.auxiliary_future_collision_weight = float(auxiliary_cfg.get("future_collision_weight", 1.0))
        self.auxiliary_future_stuck_weight = float(auxiliary_cfg.get("future_stuck_weight", 1.0))
        self.auxiliary_future_clearance_weight = float(auxiliary_cfg.get("future_clearance_weight", 1.0))
        self.auxiliary_future_progress_weight = float(auxiliary_cfg.get("future_progress_weight", 1.0))
        self.auxiliary_output_dim = 4 * len(self.auxiliary_future_horizons)
        if self.auxiliary_enabled and self.auxiliary_loss_weight > 0.0:
            auxiliary_hidden_dim = int(auxiliary_cfg.get("hidden_dim", 128))
            self.auxiliary_predictor = nn.Sequential(
                nn.Linear(256 + self.action_dim, auxiliary_hidden_dim),
                nn.ELU(),
                nn.LayerNorm(auxiliary_hidden_dim),
                nn.Linear(auxiliary_hidden_dim, self.auxiliary_output_dim),
            ).to(self.device)
        else:
            self.auxiliary_predictor = None

        behavior_cloning_cfg = cfg.feature_extractor.get("behavior_cloning", {})
        self.behavior_cloning_enabled = bool(behavior_cloning_cfg.get("enabled", False))
        self.behavior_cloning_loss_weight = float(behavior_cloning_cfg.get("loss_weight", 0.0))
        self.behavior_cloning_batch_size = max(int(behavior_cloning_cfg.get("batch_size", 256)), 1)
        self.behavior_cloning_min_buffer_size = max(
            int(behavior_cloning_cfg.get("min_buffer_size", self.behavior_cloning_batch_size)),
            1,
        )
        self.behavior_cloning_updates_per_iteration = max(
            int(behavior_cloning_cfg.get("updates_per_iteration", 8)),
            0,
        )
        self.behavior_cloning_max_log_prob = float(
            behavior_cloning_cfg.get("max_log_prob", 10.0)
        )
        self.behavior_cloning_max_nll = max(
            float(behavior_cloning_cfg.get("max_nll", 30.0)),
            1.0,
        )

        # Actor etwork
        self.actor = ProbabilisticActor(
            TensorDictModule(BetaActor(self.action_dim), ["_feature"], ["alpha", "beta"]),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")], 
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)

        # Critic network
        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"] 
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)

        # Loss related
        self.gae = GAE(0.99, 0.95) # generalized adavantage esitmation
        self.critic_loss_fn = nn.HuberLoss(delta=10) # huberloss (L1+L2): https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html

        # Optimizer
        feature_extractor_params = list(self.feature_extractor.parameters())
        if self.auxiliary_predictor is not None:
            feature_extractor_params += list(self.auxiliary_predictor.parameters())
        self.feature_extractor_params = feature_extractor_params
        self.feature_extractor_optim = torch.optim.Adam(self.feature_extractor_params, lr=cfg.feature_extractor.learning_rate)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic.learning_rate,)

        # Dummy Input for nn lazymodule
        dummy_input = observation_spec.zero()
        # print("dummy_input: ", dummy_input)


        self.__call__(dummy_input)

        # Initialize network
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        self.actor.apply(init_)
        self.critic.apply(init_)
        if self.auxiliary_predictor is not None:
            self._init_auxiliary_predictor()

    def __call__(self, tensordict):
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)

        # Cooridnate change: transform local to world
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        tensordict["agents", "action"] = actions_world
        return tensordict

    def _init_auxiliary_predictor(self):
        for module in self.auxiliary_predictor.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.1)
                nn.init.constant_(module.bias, 0.0)
        output_layer = self.auxiliary_predictor[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    @staticmethod
    def _symlog(x):
        return torch.sign(x) * torch.log1p(x.abs())

    def _auxiliary_input(self, tensordict):
        features = tensordict["_feature"]
        action = tensordict[("agents", "action_normalized")]
        action = action.reshape(*features.shape[:-1], -1)
        return torch.cat([features, action], dim=-1)

    def _future_within_horizon(self, signal, done, horizon):
        target = torch.zeros_like(signal, dtype=torch.bool)
        valid_for_offset = torch.ones_like(done, dtype=torch.bool)
        signal_bool = signal >= 0.5
        done_bool = done.bool()

        for offset in range(horizon):
            shifted_signal = torch.zeros_like(signal_bool)
            if offset == 0:
                shifted_signal = signal_bool
            else:
                prev_done = torch.ones_like(done_bool)
                prev_done[:, :-offset] = done_bool[:, offset - 1:-1]
                valid_for_offset = valid_for_offset & (~prev_done)
                shifted_signal[:, :-offset] = signal_bool[:, offset:]
            target = target | (shifted_signal & valid_for_offset)
        return target.float()

    def _multi_step_sum(self, signal, done, horizon):
        target = torch.zeros_like(signal)
        valid_for_offset = torch.ones_like(done, dtype=torch.bool)
        done_bool = done.bool()

        for offset in range(horizon):
            shifted_signal = torch.zeros_like(signal)
            if offset == 0:
                shifted_signal = signal
            else:
                prev_done = torch.ones_like(done_bool)
                prev_done[:, :-offset] = done_bool[:, offset - 1:-1]
                valid_for_offset = valid_for_offset & (~prev_done)
                shifted_signal[:, :-offset] = signal[:, offset:]
            target = target + torch.where(valid_for_offset, shifted_signal, torch.zeros_like(shifted_signal))
        return target

    def _multi_step_min(self, signal, done, horizon):
        fill_value = signal.detach().max()
        target = torch.full_like(signal, fill_value)
        valid_for_offset = torch.ones_like(done, dtype=torch.bool)
        done_bool = done.bool()

        for offset in range(horizon):
            shifted_signal = torch.full_like(signal, fill_value)
            if offset == 0:
                shifted_signal = signal
            else:
                prev_done = torch.ones_like(done_bool)
                prev_done[:, :-offset] = done_bool[:, offset - 1:-1]
                valid_for_offset = valid_for_offset & (~prev_done)
                shifted_signal[:, :-offset] = signal[:, offset:]
            target = torch.minimum(target, torch.where(valid_for_offset, shifted_signal, torch.full_like(signal, fill_value)))
        return target

    def _add_multi_step_auxiliary_targets(self, tensordict):
        if self.auxiliary_predictor is None or not self.auxiliary_future_horizons:
            return

        next_stats = tensordict["next", "stats"]
        collision = next_stats["collision"].detach().squeeze(-1)
        stuck = next_stats["stuck_active"].detach().squeeze(-1)
        front_clearance = next_stats["front_clearance"].detach().squeeze(-1)
        goal_progress = next_stats["goal_progress"].detach().squeeze(-1)
        done = tensordict["next", "done"].detach().squeeze(-1)

        future_collision = []
        future_stuck = []
        future_clearance = []
        future_progress = []
        for horizon in self.auxiliary_future_horizons:
            future_collision.append(self._future_within_horizon(collision, done, horizon))
            future_stuck.append(self._future_within_horizon(stuck, done, horizon))
            future_clearance.append(self._multi_step_min(front_clearance, done, horizon))
            future_progress.append(self._multi_step_sum(goal_progress, done, horizon))

        tensordict.set("_aux_future_collision", torch.stack(future_collision, dim=-1))
        tensordict.set("_aux_future_stuck", torch.stack(future_stuck, dim=-1))
        tensordict.set("_aux_future_clearance", torch.stack(future_clearance, dim=-1))
        tensordict.set("_aux_future_progress", torch.stack(future_progress, dim=-1))

    def _compute_auxiliary_loss(self, tensordict):
        if self.auxiliary_predictor is None:
            zero = tensordict["_feature"].sum() * 0.0
            return zero, TensorDict({
                "auxiliary_loss": zero.detach(),
                "auxiliary_future_progress_loss": zero.detach(),
                "auxiliary_future_clearance_loss": zero.detach(),
                "auxiliary_future_collision_loss": zero.detach(),
                "auxiliary_future_stuck_loss": zero.detach(),
            }, [])

        predictions = self.auxiliary_predictor(self._auxiliary_input(tensordict))
        future_predictions = predictions.reshape(
            *predictions.shape[:-1],
            len(self.auxiliary_future_horizons),
            4,
        )
        pred_future_collision = future_predictions[..., 0]
        pred_future_stuck = future_predictions[..., 1]
        pred_future_clearance = future_predictions[..., 2]
        pred_future_progress = future_predictions[..., 3]

        target_future_collision = tensordict["_aux_future_collision"].detach().clamp(0.0, 1.0)
        target_future_stuck = tensordict["_aux_future_stuck"].detach().clamp(0.0, 1.0)
        target_future_clearance = self._symlog(tensordict["_aux_future_clearance"].detach())
        target_future_progress = self._symlog(tensordict["_aux_future_progress"].detach())

        future_collision_loss = F.binary_cross_entropy_with_logits(
            pred_future_collision,
            target_future_collision,
        )
        future_stuck_loss = F.binary_cross_entropy_with_logits(
            pred_future_stuck,
            target_future_stuck,
        )
        future_clearance_loss = F.smooth_l1_loss(pred_future_clearance, target_future_clearance)
        future_progress_loss = F.smooth_l1_loss(pred_future_progress, target_future_progress)

        raw_auxiliary_loss = (
            self.auxiliary_future_collision_weight * future_collision_loss
            + self.auxiliary_future_stuck_weight * future_stuck_loss
            + self.auxiliary_future_clearance_weight * future_clearance_loss
            + self.auxiliary_future_progress_weight * future_progress_loss
        )
        auxiliary_loss = self.auxiliary_loss_weight * raw_auxiliary_loss

        return auxiliary_loss, TensorDict({
            "auxiliary_loss": auxiliary_loss.detach(),
            "auxiliary_future_progress_loss": future_progress_loss.detach(),
            "auxiliary_future_clearance_loss": future_clearance_loss.detach(),
            "auxiliary_future_collision_loss": future_collision_loss.detach(),
            "auxiliary_future_stuck_loss": future_stuck_loss.detach(),
        }, [])

    def _compute_behavior_cloning_loss(self, demonstration_batch, reference_feature):
        zero = reference_feature.sum() * 0.0
        if (
            not self.behavior_cloning_enabled
            or self.behavior_cloning_loss_weight <= 0.0
            or demonstration_batch is None
        ):
            return zero, TensorDict({
                "behavior_cloning_loss": zero.detach(),
                "behavior_cloning_nll": zero.detach(),
                "behavior_cloning_confidence": zero.detach(),
                "behavior_cloning_active": zero.detach(),
            }, [])

        self.feature_extractor(demonstration_batch)
        action_dist = self.actor.get_dist(demonstration_batch)
        target_action = demonstration_batch["teacher_action_normalized"].detach()
        confidence = demonstration_batch["teacher_confidence"].detach().clamp(0.0, 1.0)
        log_prob = action_dist.log_prob(target_action)

        # Beta densities can exceed one, while boundary targets can also produce
        # very small densities. Bound both tails to keep this auxiliary term stable.
        nll = -log_prob.clamp(
            min=-self.behavior_cloning_max_nll,
            max=self.behavior_cloning_max_log_prob,
        )
        confidence_sum = confidence.sum().clamp_min(1e-6)
        raw_loss = (confidence * nll).sum() / confidence_sum
        behavior_cloning_loss = self.behavior_cloning_loss_weight * raw_loss
        active = torch.ones((), device=behavior_cloning_loss.device)
        return behavior_cloning_loss, TensorDict({
            "behavior_cloning_loss": behavior_cloning_loss.detach(),
            "behavior_cloning_nll": raw_loss.detach(),
            "behavior_cloning_confidence": confidence.mean().detach(),
            "behavior_cloning_active": active,
        }, [])

    def train(self, tensordict, demonstration_buffer=None):
        # tensordict: (num_env, num_frames, dim), batchsize = num_env * num_frames
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_tensordict = torch.vmap(self.feature_extractor)(next_tensordict) # calculate features for next state value calculation
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict["next", "agents", "reward"] # Reward obtained by state transition
        dones = tensordict["next", "terminated"] # Whether the next states are terminal states

        values = tensordict["state_value"] # This is calculated stored when we called forward to obtain actions
        values = self.value_norm.denormalize(values) # denomalize values based on running mean and var of return
        next_values = self.value_norm.denormalize(next_values)

        # calculate GAE: Generalized Advantage Estimation
        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        self.value_norm.update(ret) # update running mean and var for return
        ret = self.value_norm.normalize(ret)  # normalize return
        tensordict.set("adv", adv)
        tensordict.set("ret", ret)
        self._add_multi_step_auxiliary_targets(tensordict)

        # Training
        infos = []
        total_updates = self.cfg.training_epoch_num * self.cfg.num_minibatches
        bc_update_count = min(self.behavior_cloning_updates_per_iteration, total_updates)
        if bc_update_count <= 0:
            bc_update_indices = set()
        elif bc_update_count == 1:
            bc_update_indices = {0}
        else:
            bc_update_indices = {
                int(round(index * (total_updates - 1) / (bc_update_count - 1)))
                for index in range(bc_update_count)
            }
        update_index = 0
        for epoch in range(self.cfg.training_epoch_num):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                demonstration_batch = None
                if (
                    update_index in bc_update_indices
                    and self.behavior_cloning_enabled
                    and demonstration_buffer is not None
                    and len(demonstration_buffer) >= self.behavior_cloning_min_buffer_size
                ):
                    demonstration_batch = demonstration_buffer.sample(
                        self.behavior_cloning_batch_size,
                        self.device,
                    )
                infos.append(self._update(minibatch, demonstration_batch))
                update_index += 1
        infos = torch.stack(infos).to_tensordict()
        
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}    

    
    def _update(self, tensordict, demonstration_batch=None): # tensordict shape (batch_size, )
        self.feature_extractor(tensordict)

        # Get action from the current policy
        action_dist = self.actor.get_dist(tensordict) # this does an actor forward to get "loc" and "scale" and use them to build multivariate normal distribution
        log_probs = action_dist.log_prob(tensordict[("agents", "action_normalized")]) # based on the gaussian, we can calculate the log prob of the action from the current policy

        # Entropy Loss
        action_entropy = action_dist.entropy()
        entropy_loss = -self.cfg.entropy_loss_coefficient * torch.mean(action_entropy)

        # Actor Loss
        advantage = tensordict["adv"] # the advantage is calculated based on GAE in hte previous step
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = advantage * ratio
        surr2 = advantage * ratio.clamp(1.-self.cfg.actor.clip_ratio, 1.+self.cfg.actor.clip_ratio)
        actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim 

        # Critic Loss 
        b_value = tensordict["state_value"]
        ret = tensordict["ret"] # Return G
        value = self.critic(tensordict)["state_value"] 
        value_clipped = b_value + (value - b_value).clamp(-self.cfg.critic.clip_ratio, self.cfg.critic.clip_ratio) # this guarantee that critic update is clamped
        critic_loss_clipped = self.critic_loss_fn(ret, value_clipped)
        critic_loss_original = self.critic_loss_fn(ret, value)
        critic_loss = torch.max(critic_loss_clipped, critic_loss_original)
        auxiliary_loss, auxiliary_info = self._compute_auxiliary_loss(tensordict)
        behavior_cloning_loss, behavior_cloning_info = self._compute_behavior_cloning_loss(
            demonstration_batch,
            tensordict["_feature"],
        )

        # Total Loss
        loss = entropy_loss + actor_loss + critic_loss + auxiliary_loss + behavior_cloning_loss

        # Optimize
        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()

        feature_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.feature_extractor_params, max_norm=5.)
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=5.) # to prevent gradient growing too large
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=5.)
        self.feature_extractor_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()
        explained_var = 1 - F.mse_loss(value, ret) / ret.var()
        info = TensorDict({
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy_loss,
            "feature_grad_norm": feature_grad_norm,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var,
        }, [])
        info.update(auxiliary_info)
        info.update(behavior_cloning_info)
        return info
