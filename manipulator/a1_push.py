import imp
from yaml import load
from legged_gym.legged_gym.envs.base.base_task import BaseTask
from legged_gym.legged_gym.envs.base.legged_robot import LeggedRobot
from manipulator.a1_config import A1Cfg, LeggedRobotCfg
import os
from legged_gym.legged_gym import LEGGED_GYM_ROOT_DIR
from isaacgym_utils.scene import GymScene
from isaacgym import gymapi

class empty:
    pass

class LeggedRobotPush(LeggedRobot):
    def __init__(self, cfg: A1Cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _create_envs(self):
        assets = self.cfg.asset.assets
        asset_collect = {}
        for asset in assets:
            asset_collect[asset] = {}
            load_asset = getattr(self.cfg.asset, asset, None)
            if load_asset is not None:
                asset_opts = load_asset.getattr('asset_options', empty()).__dict__
                gym_asset_options = gymapi.AssetOptions()
                for key, value in asset_opts.items():
                    setattr(gym_asset_options, key, value)
                
                if getattr(load_asset, "file", None) is not None:
                    # load from file
                    asset_path = load_asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
                    asset_root = os.path.dirname(asset_path)
                    asset_file = os.path.basename(asset_path)
                    asset_collect[asset]['asset'] = self.gym.load_asset(self.sim, asset_root, asset_file, gym_asset_options)

                else:
                    # load procedural asset
                    if asset == 'box':
                        asset_collect[asset]['asset'] = self.gym.create_box(self.sim, load_asset.sx, load_asset.sy, load_asset.sz, gym_asset_options)

                asset_collect[asset]['num_dof'] = self.gym.get_asset_dof_count(asset_collect[asset]['asset'])
                asset_collect[asset]['num_bodies'] = self.gym.get_asset_rigid_body_count(asset_collect[asset]['asset'])
                asset_collect[asset]['dof_props_asset'] = self.gym.get_asset_dof_properties(asset_collect[asset]['asset'])
                asset_collect[asset]['rigid_shape_props_asset'] = self.gym.get_asset_rigid_shape_properties(asset_collect[asset]['asset'])
                asset_collect[asset]['body_names'] = self.gym.get_asset_rigid_body_names(asset_collect[asset]['asset'])
                
                self.dof_names = self.gym.get_asset_dof_names(asset_collect[asset]['asset'])
                self.num_dof += asset_collect[asset]['num_dof']
                self.num_bodies += asset_collect[asset]['num_bodies']

                if getattr(load_asset, 'foot_name', None) is not None:
                    asset_collect[asset]['feet_names'] = [s for s in asset_collect[asset]['body_names'] if load_asset.foot_name in s]
                
                if getattr(load_asset, 'penalize_contacts_on', None) is not None:
                    asset_collect[asset]['penalized_contact_names'] = []
                    for name in load_asset.penalize_contacts_on:
                        asset_collect[asset]['penalized_contact_names'].extend([s for s in asset_collect[asset]['body_names'] if name in s])

                if getattr(load_asset, 'termination_contact_names', None) is not None:
                    asset_collect[asset]['termination_contact_names'] = []
                    for name in load_asset.termination_contact_names:
                        asset_collect[asset]['termination_contact_names'].extend([s for s in asset_collect[asset]['body_names'] if name in s])

                asset_collect[asset]['init_state'] = load_asset.init_state.pos + load_asset.init_state.rot + load_asset.init_state.lin_vel + load_asset.init_state.ang_vel
# class LeggedRobotPush():
#     def __init__(self, cfg):
#         scene = GymScene(cfg)


# LeggedRobotPush(LeggedRobotCfg())


# if __name__ == "__main__":
