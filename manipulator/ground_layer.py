

from isaacgym import gymapi, gymtorch
# from base_scene import BaseScene
from a1_config import SceneCfg
import numpy as np

class BaseGround():
    def __init__(self, name: str, cfg: SceneCfg.ground, scene, spacing):
        self.name = name
        self.cfg = cfg
        self.scene = scene
        # self.actor_handles = []
        self.sx, self.sy, self.sz = self.cfg.sx, self.cfg.sy, self.cfg.sz

        self.pos_z = self.sz / 2.

        assert spacing % self.sx == 0
        self.x_range = (1 * spacing) // self.sx + 1
        self.pos_x = (np.linspace(0, spacing, int(self.x_range)) + self.sx/2)[:-1]

        assert spacing % self.sy == 0
        self.y_range = (1 * spacing) // self.sy + 1
        self.pos_y = (np.linspace(0, spacing, int(self.y_range)) + self.sy/2)[:-1]

        # print(self.pos_x, self.pos_y)
        self._load_asset()

    
    def _load_asset(self):
        asset_options = gymapi.AssetOptions()
        self.robot_asset = self.scene.gym.create_box(self.scene.sim, self.sx, self.sy, self.sz, asset_options)
        self.dof_props_asset = self.scene.gym.get_asset_dof_properties(self.robot_asset)
        self.rigid_shape_props_asset = self.scene.gym.get_asset_rigid_shape_properties(self.robot_asset)


    def _create_ground_actor(self, start_pose, env_handle, env_idx, idx):
        rigid_shape_props = self._process_rigid_shape_props(self.rigid_shape_props_asset, env_idx)
        self.scene.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
        actor_handle = self.scene.gym.create_actor(env_handle, self.robot_asset, start_pose, f'{self.name}_{env_idx}_{idx}', env_idx, self.cfg.self_collisions, 0)
        dof_props = self._process_dof_props(self.dof_props_asset, env_idx)
        self.scene.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
        body_props = self.scene.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
        body_props = self._process_rigid_body_props(body_props, env_idx)
        self.scene.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
        # self.actor_handles.append(actor_handle)

        return actor_handle

    def create_ground(self, env_handle, env_idx, env_origin):
        # print(env_origin)
        ctr = 0
        # start_pose = gymapi.Transform()
        # start_pose.p = gymapi.Vec3(0, 0, self.pos_z)  + gymapi.Vec3(*env_origin)
        # actor_handle = self._create_ground_actor(start_pose, env_handle, env_idx, ctr)
        # ctr += 1
        # self.actor_handles.append(actor_handle)

        for x in self.pos_x:
            for y in self.pos_y:
                start_pose = gymapi.Transform()
                start_pose.p = gymapi.Vec3(x, y, self.pos_z)  + gymapi.Vec3(*env_origin)
                actor_handle = self._create_ground_actor(start_pose, env_handle, env_idx, ctr)
                ctr += 1
                # self.actor_handles.append(actor_handle)
        return ctr


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



