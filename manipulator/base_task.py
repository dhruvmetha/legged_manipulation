from typing import Dict, List, AnyStr, Any

from numpy import dtype

from a1_config import SceneCfg
from base_scene import BaseScene
import torch

from rsl_rl.env import VecEnv
from isaacgym import gymtorch
import time

class BaseTask(VecEnv):
    def __init__(self, scene_cfg: SceneCfg, actors: List[Dict], sim_params, physics_engine, sim_device, headless):
        self.num_envs = scene_cfg.env.num_envs
        self.num_actors = len(actors)
        self.actor_objs, self.actor_enum = [], {}
        self.scene = BaseScene(scene_cfg, sim_params, physics_engine, sim_device, headless)
        self.episode_length_buf = None
        self.reset_buf = None
        self.rew_buf = None
        self.obs_buf = None
        self.privileged_obs_buf = None
        self.max_actor_control_decimation = 1
        self.actor_control_decimation_ctr = []
        self._prepare_sim(actors)
        self._begin_sim()

        self.all_obs = torch.cumsum(torch.as_tensor([0] + [i.num_observations for i in self.actor_objs]), dim=0).to(dtype=torch.int32)
        self.all_actions = torch.cumsum(torch.as_tensor([0] + [i.num_actions for i in self.actor_objs]), dim=0).to(dtype=torch.int32)
        self.all_dofs = torch.cumsum(torch.as_tensor([0] + [i.num_dofs for i in self.actor_objs]), dim=0).to(dtype=torch.int32)
        self.all_bodies = torch.cumsum(torch.as_tensor([0] + [i.num_bodies for i in self.actor_objs]), dim=0).to(dtype=torch.int32)

        self.num_obs = self.all_obs[-1]
        self.num_privileged_obs = None
        self.num_actions = self.all_actions[-1]
        self.max_episode_length = self.scene.cfg.env.episode_length_s
        
        self._init_buffers()
    

        self.extras = {}
        self.device = self.scene.device

        self.do_nothing()


    def _prepare_sim(self, actors):
        for idx, actor in enumerate(actors):
            self.actor_objs.append(actor.get("actor_class")(actor.get("actor_cfg"), self.scene, (self.num_actors * torch.arange(self.num_envs) + idx).to(dtype=torch.int32)))
            if getattr(self.actor_objs[-1].cfg, "control", None) is not None:
                if getattr(self.actor_objs[-1].cfg.control, "decimation", None) is not None:
                    if self.max_actor_control_decimation < self.actor_objs[-1].cfg.control.decimation:
                        self.max_actor_control_decimation = self.actor_objs[-1].cfg.control.decimation
                        self.actor_control_decimation_ctr.append(self.actor_objs[-1].cfg.control.decimation)
                    else:
                        self.actor_control_decimation_ctr.append(0)
                else:
                    self.actor_control_decimation_ctr.append(0)

        self.actor_enum = dict(enumerate(self.actor_objs))

        for i in range(self.num_envs):
            self.scene.create_env()
            for actor in self.actor_objs:
                actor.create_actor(i)
                
    def get_privileged_observations(self):
        return None

    def reset(self):
        self._reset_envs(env_ids=torch.arange(self.num_envs).to(dtype=torch.long))

    def _begin_sim(self):
        self.scene.start_sim()
    
    def _init_buffers(self):
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.scene.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.scene.num_envs, device=self.scene.device, dtype=torch.bool)
        self.rew_buf = torch.zeros(self.scene.num_envs, device=self.scene.device, dtype=torch.float)
        
        actor_root_state = self.scene.gym.acquire_actor_root_state_tensor(self.scene.sim)
        dof_state_tensor = self.scene.gym.acquire_dof_state_tensor(self.scene.sim)
        net_contact_forces = self.scene.gym.acquire_net_contact_force_tensor(self.scene.sim)
        self.scene.gym.refresh_dof_state_tensor(self.scene.sim)
        self.scene.gym.refresh_actor_root_state_tensor(self.scene.sim)
        self.scene.gym.refresh_net_contact_force_tensor(self.scene.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, -1, 13) # num_envs * num_actors * root_states
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, -1, 2) # num_envs * num_all_dofs * 2
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # print(self.root_states.shape)
        # print(self.dof_state.shape)
        # print(self.contact_forces.shape)

        for idx, actor in self.actor_enum.items():
            actor.init_buffers(self.root_states[:, idx, :].contiguous().squeeze(1), self.dof_state[:, self.all_dofs[idx]: self.all_dofs[idx+1], :].contiguous(),self.contact_forces[:, self.all_bodies[idx]: self.all_bodies[idx+1], :].contiguous(), self.episode_length_buf)

    def _refresh_actors_buffer(self):
        for idx, actor in self.actor_enum.items():
            actor.refresh_buffers(self.root_states[:, idx, :].contiguous().squeeze(1), self.dof_state[:, self.all_dofs[idx]: self.all_dofs[idx+1], :].contiguous(),self.contact_forces[:, self.all_bodies[idx]: self.all_bodies[idx+1].contiguous(), :], self.episode_length_buf)


    def get_observations(self):
        return self.obs_buf

    def do_nothing(self):
        actions = torch.zeros(self.scene.num_envs, self.all_actions[-1], dtype=torch.float, device=self.scene.device, requires_grad=False)
        obs, priv_obs, rew, dones, infos = self.step(actions)

    def step(self, actions):
        actor_control_decimation = self.actor_control_decimation_ctr.copy()
        self.scene.render()
        for i in range(max(actor_control_decimation)):
            for idx, actor in enumerate(self.actor_objs):
                if actor_control_decimation[idx] > 0:
                    # TODO: allow for more kinds of control (not just on dof)
                    actor.step(actions[:, self.all_actions[idx]:self.all_actions[idx+1]])
                    actor_control_decimation[idx] -= 1
            self.scene.gym.simulate(self.scene.sim)
            if self.scene.device == 'cpu':
                self.scene.gym.fetch_results(self.scene.sim)
            self.scene.gym.refresh_dof_state_tensor(self.scene.sim)
            self._refresh_actors_buffer()

        
        self._post_physics_step()
        self._check_termination()
        self._compute_reward()
        if not torch.is_tensor(self.reset_buf):
            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            self._reset_envs(env_ids)
        
        self._compute_observations()

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def _post_physics_step(self):
        self.episode_length_buf += 1
        self.scene.gym.refresh_actor_root_state_tensor(self.scene.sim) #TODO: render in base scene
        self.scene.gym.refresh_net_contact_force_tensor(self.scene.sim) #TODO: render in base scene
        self._refresh_actors_buffer()

        for idx, actor in enumerate(self.actor_objs):
            actor.post_physics_step()
        

    def _check_termination(self):
        # TODO: implement for local actor resets, now only implemented for global sim resets.
        self.reset_buf = None
        for idx, actor in enumerate(self.actor_objs):
            if not torch.is_tensor(self.reset_buf):
                self.reset_buf = actor.reset_buf
            self.reset_buf |= actor.reset_buf.to(dtype=torch.bool)
        
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf
    
    def _compute_reward(self):
        self.rew_buf[:] = 0.
        for idx, actor in enumerate(self.actor_objs):
            self.rew_buf[:] += actor.rew_buf[:]

    def _reset_envs(self, env_ids):
        if len(env_ids) == 0:
            return
        for idx, actor in enumerate(self.actor_objs):
            actor.reset_idx(env_ids)
        
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        # self._reset_dofs(env_ids)
        # self._reset_root_states(env_ids)


    def _reset_dofs(env_ids):
        pass

    def _reset_root_states(env_ids):
        pass

    def _compute_observations(self):
        self.obs_buf = None
        for idx, actor in enumerate(self.actor_objs):
            actor.compute_observations()
            if torch.is_tensor(actor.obs_buf):
                if not torch.is_tensor(self.obs_buf):
                    self.obs_buf = actor.obs_buf
                else:
                    self.obs_buf = torch.cat([self.obs_buf, actor.obs_buf], dim=-1)
            else:
                continue

  





    
        


