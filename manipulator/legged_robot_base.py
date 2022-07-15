import os
import torch
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import torch_rand_float, to_torch, get_axis_params, quat_rotate_inverse
from math_utils import wrap_to_pi, quat_apply
from legged_gym import LEGGED_GYM_ROOT_DIR

from a1_config import LeggedRobotCfg
from base_scene import BaseScene
from utils import class_to_dict

class LeggedRobot():
    def __init__(self, name: str, cfg: LeggedRobotCfg, scene: BaseScene):
        self.name = name
        self.cfg = cfg
        self.scene = scene
        self._parse_cfg()
        self.num_observations = self.cfg.env.num_observations
        self.num_actions = self.cfg.env.num_actions
        self.decimation_rate = self.cfg.control.decimation

        self.obs_buf = torch.zeros(self.scene.num_envs, self.num_observations, device=self.scene.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.scene.num_envs, device=self.scene.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.scene.num_envs, device=self.scene.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.scene.num_envs, device=self.scene.device, dtype=torch.bool)

        self.episode_length_buf = None

        self.actor_handles = []
        self._load_asset()
        self._prepare_reward_function()

        self.extras = {}

        # self.env_ids_int32 = actor_idxs.to(self.scene.device)

    def _load_asset(self):
        
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()

        for key, value in class_to_dict(self.cfg.asset.asset_options).items():
            setattr(asset_options, key, value)
            
        
        self.robot_asset = self.scene.gym.load_asset(self.scene.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.scene.gym.get_asset_dof_count(self.robot_asset)
        self.num_bodies = self.scene.gym.get_asset_rigid_body_count(self.robot_asset)
        self.dof_props_asset = self.scene.gym.get_asset_dof_properties(self.robot_asset)
        self.rigid_shape_props_asset = self.scene.gym.get_asset_rigid_shape_properties(self.robot_asset)

        # save body names from the asset
        self.body_names = self.scene.gym.get_asset_rigid_body_names(self.robot_asset)
        self.dof_names = self.scene.gym.get_asset_dof_names(self.robot_asset)
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        
        
        self.feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        self.penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            self.penalized_contact_names.extend([s for s in self.body_names if name in s])
        self.termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            self.termination_contact_names.extend([s for s in self.body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.scene.device, requires_grad=False)
        
    
    def create_actor(self, env_handle, env_idx, env_origin):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        start_pose = gymapi.Transform()
        self.env_origins = self.scene.env_origins
        # print(env_origin)
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        # env_lower = gymapi.Vec3(0., 0., 0.)
        # env_upper = gymapi.Vec3(0., 0., 0.)
        # for i, env_handle in enumerate(self.scene.envs):
        # create env instance
        # env_handle = self.scene.gym.create_env(self.scene.sim, env_lower, env_upper, int(np.sqrt(self.scene.num_envs)))
        # env_handle = self.scene.envs[env_idx]
        pos = env_origin.clone() # self.env_origins[env_idx].clone()
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

        self.feet_indices = torch.zeros(len(self.feet_names), dtype=torch.long, device=self.scene.device, requires_grad=False)
        for i in range(len(self.feet_names)):
            self.feet_indices[i] = self.scene.gym.find_actor_rigid_body_handle(env_handle, actor_handle, self.feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(self.penalized_contact_names), dtype=torch.long, device=self.scene.device, requires_grad=False)
        for i in range(len(self.penalized_contact_names)):
            self.penalised_contact_indices[i] = self.scene.gym.find_actor_rigid_body_handle(env_handle, actor_handle, self.penalized_contact_names[i])

        # print('hereeee')
        self.termination_contact_indices = torch.zeros(len(self.termination_contact_names), dtype=torch.long, device=self.scene.device, requires_grad=False)
        # print(self.termination_contact_indices)
        for i in range(len(self.termination_contact_names)):
            self.termination_contact_indices[i] = self.scene.gym.find_actor_rigid_body_handle(env_handle, actor_handle, self.termination_contact_names[i])
        # print(self.termination_contact_indices)

        return actor_handle

    def init_buffers(self):
        
        # cloning the root states, dof, net_contact_forces for this asset to process
        self.root_states = self.scene.root_states[self.scene.actor_indices[self.name]['root']].clone().contiguous() 

        self.dof_state = self.scene.dof_state[self.scene.actor_indices[self.name]['dof_sim']].clone().contiguous()

        self.contact_forces = self.scene.contact_forces #[:, self.scene.actor_indices[self.name]['rb_env'], :].clone().contiguous()

        
        # deriving and creating more buffers
        self.dof_pos = self.dof_state.view(self.scene.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.scene.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.scene.up_axis_idx), device=self.scene.device).repeat((self.scene.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.scene.device).repeat((self.scene.num_envs, 1))
        self.torques = torch.zeros(self.scene.num_envs, self.num_actions, dtype=torch.float, device=self.scene.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.scene.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.scene.device, requires_grad=False)
        self.actions = torch.zeros(self.scene.num_envs, self.num_actions, dtype=torch.float, device=self.scene.device, requires_grad=False)
        self.last_actions = torch.zeros(self.scene.num_envs, self.num_actions, dtype=torch.float, device=self.scene.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.scene.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.scene.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.scene.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.scene.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.scene.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.scene.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.scene.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.scene.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.scene.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # other global scene buffers required for processing

        self.episode_length_buf = self.scene.episode_length_buf
        self.time_out_buf = self.scene.time_out_buf


    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.scene.device)
        # step physics and render each frame
        # self.render() #TODO: render in base scene
        # for _ in range(self.cfg.control.decimation):
        self.torques = self._compute_torques(self.actions).view(self.torques.shape)

        self.scene.dof_actuation_force[self.scene.actor_indices[self.name]['dof_sim']] = self.torques.flatten()

        # print('here')
        # TODO: should we call this in base scene.
        h = self.scene.gym.set_dof_actuation_force_tensor_indexed(self.scene.sim, gymtorch.unwrap_tensor(self.scene.dof_actuation_force), gymtorch.unwrap_tensor(torch.as_tensor(self.scene.actor_indices[self.name]['root'], dtype=torch.int32, device=self.scene.device)), len(self.scene.actor_indices[self.name]['root']))

        # print('here')


        #TODO: render in base scene
        # self.scene.gym.simulate(self.scene.sim) #TODO: render in base scene
        # if self.scene.device == 'cpu': #TODO: render in base scene
        #     self.scene.gym.fetch_results(self.scene.sim, True) #TODO: render in base scene
        # self.scene.gym.refresh_dof_state_tensor(self.scene.sim) #TODO: render in base scene
        # self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        # clip_obs = self.cfg.normalization.clip_observations
        # self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        # if self.privileged_obs_buf is not None:
        #     self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        # return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def refresh_buffers(self):
        # cloning the root states, dof, net_contact_forces for this asset to process
        self.root_states = self.scene.root_states[self.scene.actor_indices[self.name]['root']].clone().contiguous()

        self.dof_state = self.scene.dof_state[self.scene.actor_indices[self.name]['dof_sim']].clone().contiguous()

        self.contact_forces = self.scene.contact_forces # [:, self.scene.actor_indices[self.name]['rb_env'], :].clone().contiguous()

        self.dof_pos = self.dof_state.view(self.scene.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.scene.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        # other global scene buffers required for processing

        # self.episode_length_buf = args[3]
        self.time_out_buf = self.scene.time_out_buf
        self.episode_length_buf = self.scene.episode_length_buf

        

    def post_physics_step(self, *args):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        # self.gym.refresh_actor_root_state_tensor(self.sim) #TODO: render in base scene
        # self.gym.refresh_net_contact_force_tensor(self.sim) #TODO: render in base scene

        # self.episode_length_buf += 1 #TODO: render in base scene
        self.common_step_counter += 1 
        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # self.reset_idx(env_ids)
        # self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # if self.viewer and self.enable_viewer_sync and self.debug_viz:
        #     self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """

        # print(self.termination_contact_indices, self.contact_forces.shape)
        # print(self.contact_forces)
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # self.time_out_buf = self.episode_length_buf > self.max_episode_length #TODO: render in base scene # no terminal reward for time-outs
        # self.reset_buf |= self.time_out_buf #TODO: render in base scene


    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            
            rew = self.reward_functions[i]() * self.reward_scales[name]
            # print(name, rew)
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers
        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        # if self.cfg.terrain.curriculum:
        #     self._update_terrain_curriculum(env_ids)
        # # avoid updating command curriculum at each step since the maximum command is common to all envs
        # if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
        #     self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        # if self.cfg.terrain.curriculum:
        #     self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        # if self.cfg.commands.curriculum:
        #     self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf


    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # TODO: remove noise for inference (model_id)
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.scene.device)
        self.dof_vel[env_ids] = 0.
        
        # env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.scene.dof_state[self.scene.actor_indices[self.name]['dof_sim']] = self.dof_state.clone()
        
        self.scene.gym.set_dof_state_tensor_indexed(self.scene.sim,
                                              gymtorch.unwrap_tensor(self.scene.dof_state),
                                              gymtorch.unwrap_tensor(to_torch(self.scene.actor_indices[self.name]['root'], device=self.scene.device, dtype=torch.int32)), self.scene.num_envs)
    
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        # if self.custom_origins:
        #     self.root_states[env_ids] = self.base_init_state
        #     self.root_states[env_ids, :3] += self.env_origins[env_ids]
        #     self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.scene.device) # xy position within 1m of the center
        # else:
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.scene.device) # [7:10]: lin vel, [10:13]: ang vel

        # env_ids_int32 = env_ids.to(dtype=torch.int32)
        
        self.scene.root_states[self.scene.actor_indices[self.name]['root']] = self.root_states.clone()
        
        self.scene.gym.set_actor_root_state_tensor_indexed(self.scene.sim,gymtorch.unwrap_tensor(self.scene.root_states),gymtorch.unwrap_tensor(to_torch(self.scene.actor_indices[self.name]['root'], device=self.scene.device, dtype=torch.int32)), self.scene.num_envs)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.scene.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.scene.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.scene.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.scene.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
    
    
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.scene.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        # if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
        #     self._push_robots()
    
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions
        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)


    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.scene.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        # add noise if needed
        # if self.add_noise:
        #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

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
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.scene.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
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
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.scene.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.scene.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.scene.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
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

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)
        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.scene.cfg.terrain.measured_points_y, device=self.scene.device, requires_grad=False)
        x = torch.tensor(self.scene.cfg.terrain.measured_points_x, device=self.scene.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.scene.num_envs, self.num_height_points, 3, device=self.scene.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw
        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.
        Raises:
            NameError: [description]
        Returns:
            [type]: [description]
        """
        if self.scene.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.scene.num_envs, self.num_height_points, device=self.scene.device, requires_grad=False)
        elif self.scene.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        # if env_ids:
        #     points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        # else:
        #     points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        # points += self.scene.cfg.terrain.border_size
        # points = (points/self.scene.cfg.terrain.horizontal_scale).long()
        # px = points[:, :, 0].view(-1)
        # py = points[:, :, 1].view(-1)
        # px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        # py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        # heights1 = self.height_samples[px, py]
        # heights2 = self.height_samples[px+1, py]
        # heights3 = self.height_samples[px, py+1]
        # heights = torch.min(heights1, heights2)
        # heights = torch.min(heights, heights3)

        # return heights.view(self.scene.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure
        Args:
            cfg (Dict): Environment config file
        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions
        if self.scene.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec



    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.scene.num_envs, dtype=torch.float, device=self.scene.device, requires_grad=False)
                             for name in self.reward_scales.keys()}



    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)