import os
import torch
from base_scene import BaseScene
from a1_config import BoxCfg
from isaacgym import gymapi, gymtorch
from isaacgym_utils.math_utils import np_to_vec3
from isaacgym.torch_utils import to_torch, torch_rand_float
from utils import class_to_dict


class BoxBase():
    def __init__(self, name:str,cfg: BoxCfg, scene: BaseScene):
        self.name = name
        self.cfg = cfg
        self.scene = scene
        # self._parse_cfg()
        self.num_observations = self.cfg.env.num_observations
        self.num_actions = self.cfg.env.num_actions
        
        self.obs_buf = torch.zeros(self.scene.num_envs, self.num_observations, device=self.scene.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.scene.num_envs, device=self.scene.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.scene.num_envs, device=self.scene.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.scene.num_envs, device=self.scene.device, dtype=torch.bool)
        
        self.actor_handles = []

        self._load_asset()

    def _load_asset(self):
        asset_options = gymapi.AssetOptions()
        for key, value in class_to_dict(self.cfg.asset.asset_options).items():
            setattr(asset_options, key, value)

        self.robot_asset = self.scene.gym.create_box(self.scene.sim, self.cfg.asset.sx, self.cfg.asset.sy, self.cfg.asset.sz, asset_options)
        self.num_dof = self.scene.gym.get_asset_dof_count(self.robot_asset)
        self.num_bodies = self.scene.gym.get_asset_rigid_body_count(self.robot_asset)
        self.dof_props_asset = self.scene.gym.get_asset_dof_properties(self.robot_asset)
        self.rigid_shape_props_asset = self.scene.gym.get_asset_rigid_shape_properties(self.robot_asset)

        self.body_names = self.scene.gym.get_asset_rigid_body_names(self.robot_asset)
        self.dof_names = self.scene.gym.get_asset_dof_names(self.robot_asset)
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.scene.device, requires_grad=False)
        

    def create_actor(self, env_handle, env_idx, env_origin):
        start_pose = gymapi.Transform()
        self.env_origins = self.scene.env_origins
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        # env_handle = self.scene.envs[env_idx]
        pos = env_origin.clone()
        # pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.scene.device).squeeze(1)
        start_pose.p += gymapi.Vec3(*pos) 
            
        rigid_shape_props = self._process_rigid_shape_props(self.rigid_shape_props_asset, env_idx)
        self.scene.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
        actor_handle = self.scene.gym.create_actor(env_handle, self.robot_asset, start_pose, self.cfg.asset.name, env_idx, self.cfg.asset.self_collisions, 0)
        dof_props = self._process_dof_props(self.dof_props_asset, env_idx)
        self.scene.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
        body_props = self.scene.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
        body_props = self._process_rigid_body_props(body_props, env_idx)
        self.scene.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
        self.actor_handles.append(actor_handle)

        return actor_handle

    def init_buffers(self, *args):
        self.root_states = self.scene.root_states[self.scene.actor_indices[self.name]['root']].clone().contiguous()

        self.rb_states = self.scene.rb_states[self.scene.actor_indices[self.name]['rb_sim']].clone().contiguous()

        self.extras = {}

    def refresh_buffers(self, *args):
        self.root_states = self.scene.root_states[self.scene.actor_indices[self.name]['root']].clone().contiguous()

        self.rb_states = self.scene.rb_states[self.scene.actor_indices[self.name]['rb_sim']].clone().contiguous()

    def step(self, actions):
        # self.apply_force()
        self.compute_observations()
        self._compute_rewards()

    def post_physics_step(self):
        pass

    def reset_idx(self, env_ids):
        return

    def compute_observations(self):
        pass

    def _compute_rewards(self):
        pass

    def apply_force(self):
        

        pos = self.rb_states[self.scene.actor_indices[self.name]['rb_sim'], :3]

        vel = self.rb_states[self.scene.actor_indices[self.name]['rb_sim'], 7:10]

        # print(pos)
        
        force = torch.as_tensor([2.0, 0, 0], device=self.scene.device).float()

        self.scene.rigid_body_force_at_pos_tensor[:, self.scene.actor_indices[self.name]['rb_sim'][0], :] = force


        # print(self.scene.rigid_body_force_at_pos_tensor[torch.linalg.norm(vel, dim=-1) > 0.1])

        self.scene.rigid_body_force_at_pos_tensor[torch.linalg.norm(vel, dim=-1) > 0.2, self.scene.actor_indices[self.name]['rb_sim'][0], :] = torch.as_tensor([-0.0, 0, 0], device=self.scene.device).float()


        self.scene.gym.apply_rigid_body_force_at_pos_tensors(self.scene.sim, gymtorch.unwrap_tensor(self.scene.rigid_body_force_at_pos_tensor),gymtorch.unwrap_tensor(pos),gymapi.ENV_SPACE)






    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment
        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id
        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        # if self.cfg.domain_rand.randomize_friction:
        #     if env_id==0:
        #         # prepare friction randomization
        #         friction_range = self.cfg.domain_rand.friction_range
        #         num_buckets = 64
        #         bucket_ids = torch.randint(0, num_buckets, (self.scene.num_envs, 1))
        #         friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
        #         self.friction_coeffs = friction_buckets[bucket_ids]

        for s in range(len(props)):
            props[s].friction = .5 # 1 # 0.2 # self.friction_coeffs[env_id]

        # print(props[0].friction, props[0].restitution, props[0].rolling_friction)
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF
        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id
        Returns:
            [numpy.array]: Modified DOF properties
        # """
        # if env_id==0:
        #     self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.scene.device, requires_grad=False)
        #     self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.scene.device, requires_grad=False)
        #     self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.scene.device, requires_grad=False)
        #     for i in range(len(props)):
        #         self.dof_pos_limits[i, 0] = props["lower"][i].item()
        #         self.dof_pos_limits[i, 1] = props["upper"][i].item()
        #         self.dof_vel_limits[i] = props["velocity"][i].item()
        #         self.torque_limits[i] = props["effort"][i].item()
        #         # soft limits
        #         m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
        #         r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
        #         self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        #         self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        # if self.cfg.domain_rand.randomize_base_mass:
        #     rng = self.cfg.domain_rand.added_mass_range
        #     props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _parse_cfg(self):
        self.dt = self.cfg.control.decimation * self.scene.cfg.sim.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.scene.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.scene.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.scene.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)