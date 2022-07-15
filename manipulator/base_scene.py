# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import sys
from isaacgym import gymapi, gymutil, gymtorch
import numpy as np
from rsl_rl.env.vec_env import VecEnv
import torch

from a1_config import SceneCfg
from ground_layer import BaseGround
from utils import class_to_dict


class BaseScene(VecEnv):
    """
    Constructs the scene in Isaac Gym.
    Actors are created within and attached to this scene, this supports only one asset(robot) actuation, all others need to be unactuated. Use for manipulation tasks.
    """

    def __init__(self, cfg: SceneCfg, actors: list, sim_params, physics_engine, sim_device, headless):
        """
        initializes the scene in which the actors are created
        """
        self.gym = gymapi.acquire_gym()
        self.cfg = cfg
        self.sim_params = self._parse_sim_params(class_to_dict(cfg.sim), sim_params)
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'
        
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = self.cfg.env.num_envs

        # track envs
        self.envs = []

        # track assets
        self.actors = actors
        self.actor_objs = {}
        self.actor_handles = {}
        self.actor_indices = {} # tracking actor indices


        # control rates
        self.decimation_rate = 1
        self.dt = self.cfg.sim.dt
        self.max_episode_length_s = 1000
        self.max_episode_length = None


        # tracking states, actuation, updates
        self._tensors = {}
        self._actor_idxs_to_update = {}


        self.episode_length_buf = None
        self.reset_buf = None
        self.rew_buf = None
        self.obs_buf = None
        self.privileged_obs_buf = None


        self.num_obs = 0
        self.num_privileged_obs = None
        self.num_actions = 0

        self.privileged_obs = None
        
        # self.num_obs = cfg.env.num_observations
        # self.num_privileged_obs = cfg.env.num_privileged_obs
        # self.num_actions = cfg.env.num_actions


        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        

        self.create_sim()
        
        self.enable_viewer_sync = True
        self.viewer = None

        self.env_origins = None
        # self._get_env_origins()
        # self.create_env()

        self.extras = {}


        

    def start_sim(self):
        self.gym.prepare_sim(self.sim)
        if not self.headless:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """

        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        
        mesh_type = self.cfg.terrain.mesh_type
        self._create_ground_plane()

        # if mesh_type in ['heightfield', 'trimesh']:
        #     self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        # if mesh_type=='plane':
        #     self._create_ground_plane()
        # elif mesh_type=='heightfield':
        #     self._create_heightfield()
        # elif mesh_type=='trimesh':
        #     self._create_trimesh()
        # elif mesh_type is not None:
        #     raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        
    

    # def reset_idx(self):
    #     pass
        
    def reset(self):
        """ Reset all robots"""
        self._reset_envs(torch.arange(self.num_envs, dtype=torch.long, device=self.device))
        self.episode_length_buf[:] = 0.
        self.reset_buf[:] = 0.
        self.time_out_buf[:] = 0.
        return self.obs_buf, self.extras
    #     self.reset_idx(torch.arange(self.num_envs, device=self.device))
    #     obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
    #     return obs, privileged_obs

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)


    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    
    def create_env(self):
        
        spacing = self.cfg.env.env_spacing

        # loading the base ground
        base_ground = BaseGround('ground', self.cfg.ground, self, spacing)

        # loading all the assets
        for idx, actor in enumerate(self.actors):
            actor_name = actor.get('name')
            if actor_name in self.actor_objs:
                raise 'Loading assets with the same name'
            self.actor_objs[actor_name] = actor.get("actor_class")(actor_name, actor.get("actor_cfg"), self)
            self.actor_handles[actor_name] = []
            self.num_obs += self.actor_objs[actor_name].num_observations

            # using the assumption that we only have one actuated asset (robot), we use to setup dt, max_episode_length
            self.max_episode_length_s = self.cfg.env.episode_length_s
            if self.actor_objs[actor_name].num_dof > 0:
                self.decimation_rate = self.actor_objs[actor_name].decimation_rate
                self.dt *= self.decimation_rate
                self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        if self.max_episode_length is None:
                self.max_episode_length = self.max_episode_length_s
        

        # saving slice indices per asset for easy query into observations/actions for each asset
        self.all_obs = torch.cumsum(torch.as_tensor([0] + [i.num_observations for _, i in self.actor_objs.items()]), dim=0).to(dtype=torch.int32)
        self.all_actions = torch.cumsum(torch.as_tensor([0] + [i.num_actions for _, i in self.actor_objs.items()]), dim=0).to(dtype=torch.int32)

        # TODO: change this way of defining all actions
        self.num_actions = self.all_actions[-1]


        

        # creating the envs and their actors (realized assets).
        self._get_env_origins()
        current_actor_ctr = 0
        for env_idx in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, gymapi.Vec3(-0, spacing, 0.), gymapi.Vec3(-0, spacing, 0.), int(np.sqrt(self.num_envs)))

            # creating ground
            current_actor_ctr += base_ground.create_ground(env_handle, env_idx, self.env_origins[env_idx])

            for actor_name, asset in self.actor_objs.items():
                actor_handle = asset.create_actor(env_handle, env_idx, self.env_origins[env_idx])
                if actor_name not in self.actor_indices.keys():
                    self.actor_indices[actor_name] = {
                        'root' : [],
                        'rb_env': [],
                        'rb_sim': [],
                        'dof_env': [],
                        'dof_sim': []
                    }
                
                # storing actor indices for root state usage 
                self.actor_indices[actor_name]['root'].append(current_actor_ctr)
                current_actor_ctr += 1

                # storing actor indices for rigid body state usage
                # env
                self.actor_indices[actor_name]['rb_env'].extend([self.gym.get_actor_rigid_body_index(env_handle, actor_handle, rb_idx, gymapi.DOMAIN_ENV) for rb_idx in range(asset.num_bodies)])
                # sim
                self.actor_indices[actor_name]['rb_sim'].extend([self.gym.get_actor_rigid_body_index(env_handle, actor_handle, rb_idx, gymapi.DOMAIN_SIM) for rb_idx in range(asset.num_bodies)])
                

                # storing actor indices for dof state usage
                # env
                self.actor_indices[actor_name]['dof_env'].extend([self.gym.get_actor_dof_index(env_handle, actor_handle, dof_idx, gymapi.DOMAIN_ENV) for dof_idx in range(asset.num_dof)])
                # sim
                self.actor_indices[actor_name]['dof_sim'].extend([self.gym.get_actor_dof_index(env_handle, actor_handle, dof_idx, gymapi.DOMAIN_SIM) for dof_idx in range(asset.num_dof)])

                # storing actor handles for functional usage
                self.actor_handles[actor_name].append(actor_handle)
                
            # storing env handles for functional usage
            self.envs.append(env_handle)
        # self.actor_indices_tensor = {}
        # for name, vals in self.actor_indices.items():
        #     self.actor_indices_tensor[name] = {}
        #     for val, l in vals.items():
        #         self.actor_indices_tensor[name][val] = torch.as_tensor(l, dtype=torch.long, device=self.device)
        
        # self.actor_indices = self.actor_indices_tensor

    def init_buffers(self):

        # setting up data collection buffers for tracking scene rewards, dones, episode lengths, observations in every env.

        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # only one

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None


        # setting up simulation states of assets in every env.
        
        self.num_dofs_sim = self.gym.get_sim_dof_count(self.sim)
        self.num_rbs_sim = self.gym.get_sim_rigid_body_count(self.sim)
        assert self.num_rbs_sim % self.num_envs == 0
        self.num_rbs_env = self.num_rbs_sim // self.num_envs

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        rb_states_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)


        # contacts = self.gym.get_rigid_contacts(self.sim)
        # print(contacts)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)[:, ] #  num_actors * root_states
        self.rb_states = gymtorch.wrap_tensor(rb_states_tensor)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor) # num_all_dofs * 2
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        print('this is rb state', self.rb_states.shape)
        print('this is root state', self.root_states.shape)

        # additional buffers required to set forces on the assets for the physics sim or direct resets.

        self.dof_actuation_force = torch.zeros(self.num_dofs_sim, device=self.device, dtype=torch.float)

        self.rigid_body_force_at_pos_tensor = torch.zeros((self.num_envs, self.num_rbs_env, 3), device=self.device, dtype=torch.float)


        # using the above set buffers to initialize buffers for the assets

        for name, asset in self.actor_objs.items():
            asset.init_buffers()
    
    def do_nothing(self):
        actions = torch.zeros(self.num_envs, self.all_actions[-1], dtype=torch.float, device=self.device, requires_grad=False)
        # obs, priv_obs, rew, dones, infos = self.step(actions)
        self.step(actions)

    def do_random(self):
        actions = torch.rand(self.num_envs, self.all_actions[-1], dtype=torch.float, device=self.device, requires_grad=False)
        # obs, priv_obs, rew, dones, infos = self.step(actions)
        self.step(actions)


    def step(self, actions):
        self.render()
        for _ in range(self.decimation_rate):
            for idx, (name, asset) in enumerate(self.actor_objs.items()):
                if True or asset.num_dofs > 0:
                    asset.step(actions.clone().contiguous())
                    self.gym.simulate(self.sim)
                    # if self.device == 'cpu':
                    #     self.gym.fetch_results(self.sim)
                    self.gym.refresh_dof_state_tensor(self.sim)
                    self.extras = asset.extras
                asset.refresh_buffers()
        self._post_physics_step()
        self._check_termination()
        self._compute_reward()


        if torch.is_tensor(self.reset_buf):

            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

            if len(env_ids) > 0:
                # print(env_ids)
                self._reset_envs(env_ids)

        self._compute_observations()

        # print(self.extras)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def _post_physics_step(self):
        """
        housekeeping and refreshing for fresh state vectors to compute rewards/observations/dones and other info. # TODO: this should probably be in task
        """
        self.episode_length_buf += 1
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        for _, asset in self.actor_objs.items():
            asset.refresh_buffers()
        for _, asset in self.actor_objs.items():
            asset.post_physics_step()

        # print(self.contact_forces)
    
    def _check_termination(self):
        """
        checking reset buffers to check for terminated envs to line them up for a reset. # TODO: this should probably be in task
        """
        # TODO: implement for local actor resets, now only implemented for global sim resets.
        self.reset_buf = None
        for _, asset in self.actor_objs.items():
            # if asset.num_dof > 0:
            if not torch.is_tensor(self.reset_buf):
                self.reset_buf = asset.reset_buf
            self.reset_buf |= asset.reset_buf.to(dtype=torch.bool)
        
        if self.reset_buf is None:
            self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def _compute_reward(self):
        """
        collecting rewards from all assets # TODO: this should probably be in task
        """
        self.rew_buf[:] = 0.
        for _, asset in self.actor_objs.items():
            self.rew_buf[:] += asset.rew_buf[:]

        # print(torch.sum(self.rew_buf[:], dim=-1))

    
    def _reset_envs(self, env_ids):
        """
        using the env_ids, here we reset those particular environments to their root state (with/without randomness)
        """
        if len(env_ids) == 0:
            return
        print('here')
        for _, asset in self.actor_objs.items():
            asset.reset_idx(env_ids)

        self.episode_length_buf[:] = 0.
        self.reset_buf[:] = 0.
        self.time_out_buf[:] = 0.

    
    def _compute_observations(self):
        """
        collecting observations from each asset and concatenating them for usage.
        """
        self.obs_buf = None
        for _, asset in self.actor_objs.items():
            asset.compute_observations()
            if torch.is_tensor(asset.obs_buf):
                if not torch.is_tensor(self.obs_buf):
                    self.obs_buf = asset.obs_buf
                else:
                    self.obs_buf = torch.cat([self.obs_buf, asset.obs_buf], dim=-1)
            else:
                continue
        # print(self.contact_forces.shape, self.contact_forces)
        

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)


    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        # if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        #     self.custom_origins = True
        #     self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        #     # put robots at the origins defined by the terrain
        #     max_init_level = self.cfg.terrain.max_init_terrain_level
        #     if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
        #     self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
        #     self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
        #     self.max_terrain_level = self.cfg.terrain.num_rows
        #     self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
        #     self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        # else:
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    

    def _parse_sim_params(self, cfg, args):
        # code from Isaac Gym Preview 2
        # initialize sim params
        sim_params = gymapi.SimParams()
        # set some values from args
        if args.physics_engine == gymapi.SIM_FLEX:
            if args.device != "cpu":
                print("WARNING: Using Flex with GPU instead of PHYSX!")
        elif args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.use_gpu = args.use_gpu
            sim_params.physx.num_subscenes = args.subscenes
        sim_params.use_gpu_pipeline = args.use_gpu_pipeline

        # if sim options are provided in cfg, parse them and update/override above:
        # if "sim" in cfg:
        gymutil.parse_sim_config(cfg, sim_params)

        # Override num_threads if passed on the command line
        if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
            sim_params.physx.num_threads = args.num_threads

        return sim_params

    


    