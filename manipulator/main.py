import argparse
from isaacgym import gymutil
from a1_config import BoxCfg, LeggedRobotCfg, SceneCfg

from base_task import BaseTask
from legged_robot_base import LeggedRobot
from box_base import BoxBase
from base_scene import BaseScene
from train_config import BaseTaskPPO
from utils import class_to_dict, get_args

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

import time

if __name__ == "__main__":
    args = get_args()

    actors = []
    actors.append({
        "name": 'a1',
        "actor_class": LeggedRobot,
        "actor_cfg": LeggedRobotCfg
    })
    # actors.append({
    #     "name": 'block',
    #     "actor_class": BoxBase,
    #     "actor_cfg": BoxCfg
    # })

    # while True:

    scene = BaseScene(SceneCfg(), actors, args, args.physics_engine, args.sim_device, args.headless)
    
    scene.create_env()
    scene.start_sim()
    scene.init_buffers()

    while True:
        # time.sleep(1/60)
        scene.do_nothing()

    # train_config = BaseTaskPPO()
    # train_config.runner.resume = True
    # train_config_dict = class_to_dict(train_config)
    # # task = BaseScene(scene, actors, args, args.physics_engine, args.sim_device, args.headless)
    # runner = OnPolicyRunner(scene, train_config_dict, "test", device=args.rl_device)
    
    # # # runner.learn(1500)
    # runner.load('test/model_1500.pt')
    # policy = runner.get_inference_policy(device=scene.device)
    # obs = scene.get_observations()
    # for i in range(10 * int(scene.max_episode_length)):
    #     actions = policy(obs.detach())
    #     obs, _, rews, dones, infos = scene.step(actions.detach())
    # time.sleep(1/60)
    # # task.do_nothing()
    # # task.do_nothing()
    # print(task.get_observations().shape)